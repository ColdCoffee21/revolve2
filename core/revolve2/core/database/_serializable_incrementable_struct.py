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
    get_args,
    get_origin,
    get_type_hints,
)

from sqlalchemy import Column, Float, Integer, MetaData, String, Table, func
from sqlalchemy.ext.asyncio import AsyncConnection
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.future import select

from ._serializable import Serializable

Self = TypeVar("Self", bound="SerializableIncrementableStruct")


class SerializableIncrementableStruct(Serializable):
    """
    Similar to SerializableStruct, but the database object is identified by an item id and not a row id, which is shared between multiple rows which again can be identied by a version numner.

    Every time an object with the same item_id is saved its version_id is incremented.
    An extra function is added to retrieve an object using its row id which uniquely identifies a row as with a normal `SerializableStruct`.
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

    item_id: Optional[int] = None
    version: Optional[int] = None

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
            Column(
                "item_id",
                Integer,
                nullable=False,
            ),
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

        next_item_id: int = (
            await ses.execute(select(func.coalesce(func.max(cls.table.item_id), 0) + 1))
        ).scalar_one()

        for object in objects:
            if object.item_id is None:
                object.item_id = next_item_id
                next_item_id += 1

        versions: List[int] = [
            (
                await ses.execute(
                    select(
                        func.coalesce(
                            func.max(cls.table.version).filter(
                                cls.table.item_id == o.item_id
                            ),
                            0,
                        )
                        + 1
                    )
                )
            ).scalar_one()
            for o in objects
        ]

        for object, version in zip(objects, versions):
            object.version = version

        rows = [
            cls.table(
                item_id=o.item_id,
                version=version,
                **{name: val[i] for name, val in args.items()},
            )
            for i, o in enumerate(objects)
        ]

        ses.add_all(rows)

        return [object.item_id for object in objects]  # type: ignore # we just set them so they can't be none

    @classmethod
    async def from_db_by_id(
        cls: Type[Self], ses: AsyncSession, id: int
    ) -> Optional[Self]:
        """
        Deserialize this object from a database.

        If id does not exist, returns None.

        :param ses: Database session.
        :param id: Id of the object in the database.
        :returns: The deserialized object or None if id does not exist.
        """
        row = (
            await ses.execute(select(cls.table).filter(cls.table.id == id))
        ).scalar_one_or_none()

        if row is None:
            return None

        object = cls(
            **{
                col.name: (await col.type.from_db(ses, getattr(row, col.name)))
                if issubclass(col.type, Serializable)
                else getattr(row, col.name)
                for col in cls._columns
            },
        )
        object.item_id = int(row.item_id)
        object.version = int(row.version)

        return object

    @classmethod
    async def from_db(
        cls: Type[Self], ses: AsyncSession, item_id: int
    ) -> Optional[Self]:
        """
        Deserialize the latest version of this object with the the given item id.

        If no object exists, returns None.

        :param ses: Database session.
        :param item_id: Item id of the object.
        :returns: The deserialized object or None if no object with the item id exists.
        """
        row = (
            await ses.execute(
                select(cls.table)
                .filter(cls.table.item_id == item_id)
                .order_by(cls.table.version.desc())
                .limit(1)
            )
        ).scalar_one_or_none()

        if row is None:
            return None

        object = cls(
            **{
                col.name: (await col.type.from_db(ses, getattr(row, col.name)))
                if issubclass(col.type, Serializable)
                else getattr(row, col.name)
                for col in cls._columns
            },
        )
        object.item_id = int(row.item_id)
        object.version = int(row.version)

        return object