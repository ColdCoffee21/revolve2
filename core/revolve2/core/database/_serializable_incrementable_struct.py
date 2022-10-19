from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from sqlalchemy import Column, Float, Integer, MetaData, String, Table
from sqlalchemy.ext.asyncio import AsyncConnection
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.future import select

from ._serializable import Serializable

Self = TypeVar("Self", bound="SerializableIncrementableStruct")


class SerializableIncrementableStruct(Serializable):
    """
    Similar to SerializableStruct, but the database object can be identified by both its id and the tuple of item_id and version_id.

    Everytime an object with the same item_id is saved its version_id is incremented.
    An extra function is added to retrieve the object with an item_id and the latest version_id.
    """

    @dataclass
    class __Column:
        name: str
        type: Union[
            Type[int],
            Type[float],
            Type[str],
            Type[Serializable],
            Type[Optional[int]],
            Type[Optional[float]],
            Type[Optional[str]],
            Type[Optional[Serializable]],
        ]

    _columns: List[__Column]
    __db_base: Any  # TODO type
    __db_id: Optional[int] = None

    @classmethod
    def __init_subclass__(
        cls: Type[Self],
        /,
        table_name: str,
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Initialize this object.

        :param table_name: Name of table in database. If None this function will be skipped, which is for internal use only.
        :param kwargs: Other arguments not specific to this class.
        """
        super().__init_subclass__(**kwargs)

        base_keys = get_type_hints(SerializableIncrementableStruct).keys()
        columns = [
            cls.__make_column(name, ctype)
            for name, ctype in get_type_hints(cls).items()
            if name not in base_keys
        ]
        cls._columns = [col[0] for col in columns]

        metadata = MetaData()
        Table(
            table_name,
            metadata,
            Column(
                "id",
                Integer,
                nullable=False,
                primary_key=True,
                unique=True,
                autoincrement=True,
            ),
            Column("item_id", Integer, nullable=False),
            Column("version", Integer, nullable=False),
            *[
                Column(col.name, cls.__sqlalchemy_type(col.type), nullable=nullable)
                for (col, nullable) in columns
            ],
        )
        cls.__db_base = automap_base(metadata=metadata)
        cls.__db_base.prepare()
        cls.table = cls.__db_base.classes[table_name]

    @classmethod
    def __make_column(cls: Type[Self], name: str, ctype: Any) -> Tuple[__Column, bool]:
        if get_origin(ctype) == Union:
            args = get_args(ctype)
            assert (
                len(args) == 2
            ), "All member types must be one of 'int', 'float', 'str', 'Serializable', 'Optional[int]', 'Optional[float]', 'Optional[str]', 'Optional[Serializable]'"
            if args[0] is type(None):
                actual_type = args[1]
            elif args[1] is type(None):
                actual_type = args[0]
            else:
                assert (
                    False
                ), "All member types must be one of 'int', 'float', 'str', 'Serializable', 'Optional[int]', 'Optional[float]', 'Optional[str]', 'Optional[Serializable]'"
            nullable = True
        else:
            actual_type = ctype
            nullable = False

        assert (
            issubclass(actual_type, Serializable)
            or actual_type == int
            or actual_type == float
            or actual_type == str
        ), "All member types must be one of 'int', 'float', 'str', 'Serializable', 'Optional[int]', 'Optional[float]', 'Optional[str]', 'Optional[Serializable]'"

        return (cls.__Column(name, actual_type), nullable)

    def __sqlalchemy_type(
        ctype: Any,
    ) -> Union[Type[Integer], Type[Float], Type[String]]:
        if issubclass(ctype, Serializable):
            return Integer
        elif ctype == int:
            return Integer
        elif ctype == float:
            return Float
        elif ctype == String:
            return String
        else:
            assert False, "Unsupported type."

    @classmethod
    async def prepare_db(cls: Type[Self], conn: AsyncConnection) -> None:
        """
        Set up the database, creating tables.

        :param conn: Connection to the database.
        """
        for col in cls._columns:
            if issubclass(col.type, Serializable):
                await col.type.prepare_db(conn)
        await conn.run_sync(cls.__db_base.metadata.create_all)

    @classmethod
    async def to_db_multiple(
        cls: Type[Self], ses: AsyncSession, objects: List[Self]
    ) -> List[int]:
        """
        Serialize multiple objects to a database.

        :param ses: Database session.
        :param objects: The objects to serialize.
        :returns: Ids of the objects in the database.
        """
        args: Dict[str, Union[List[int], List[float], List[str]]] = {}

        for col in cls._columns:
            if issubclass(col.type, Serializable):
                args[col.name] = await col.type.to_db_multiple(
                    ses, [getattr(o, col.name) for o in objects]
                )
            else:
                args[col.name] = [getattr(o, col.name) for o in objects]

        rows = [
            cls.table(
                item_id=0,
                version=0,
                **{name: val[i] for name, val in args.items()},
            )
            for i in range(len(objects))
        ]

        ses.add_all(rows)
        await ses.flush()
        return [int(row.id) for row in rows]

    @classmethod
    async def from_db(cls: Type[Self], ses: AsyncSession, id: int) -> Optional[Self]:
        """
        Deserialize this object from a database.

        If id does not exist, returns None.

        :param ses: Database session.
        :param id: Id of the object in the database.
        :returns: The deserialized object or None is id does not exist.
        """
        return None  # TODO
        row = (
            await ses.execute(select(cls.table).filter(cls.table.id == id))
        ).scalar_one_or_none()

        return cls(
            **{
                col.name: (await col.type.from_db(ses, getattr(row, col.name)))
                if issubclass(col.type, Serializable)
                else getattr(row, col.name)
                for col in cls._columns
            }
        )
