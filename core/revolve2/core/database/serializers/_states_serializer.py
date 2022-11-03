from __future__ import annotations

from dataclasses import dataclass
from typing import List

from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.future import select
from typing import List, Optional, Tuple
from revolve2.core.database import IncompatibleError, Serializer

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String

from revolve2.core.physics.running import (
    ActorState
)
import pprint
import sys

DbBase = declarative_base()


class DbStates(DbBase):
    __tablename__ = "states"

    id = Column(
        Integer,
        nullable=False,
        unique=True,
        autoincrement=True,
        primary_key=True,
    )
    serialized_states = Column(String, nullable=False)


class StatesSerializer(Serializer[List[Tuple[float, ActorState]]]):
    @classmethod
    async def create_tables(cls, session: AsyncSession) -> None:
        await (await session.connection()).run_sync(DbBase.metadata.create_all)

    @classmethod
    def identifying_table(cls) -> str:
        return DbStates.__tablename__

    @classmethod
    async def to_database(
        cls, session: AsyncSession, objects: List[Tuple[float, ActorState]]
    ) -> List[int]:
        dbstates = [
            DbStates(serialized_states=str(o))
            for o in objects
        ]

        session.add_all(dbstates)
        await session.flush()

        ids = [
            dbstate.id for dbstate in dbstates
        ]  # cannot be none because not nullable. used to silence mypy
        assert len(ids) == len(objects)  # but check just to be sure

        return ids

    @classmethod
    async def from_database(
        cls, session: AsyncSession, ids: List[int]
    ) -> List[Genotype]:
        rows = (
            (await session.execute(select(DbStates).filter(DbStates.id.in_(ids))))
            .scalars()
            .all()
        )

        states = [s.serialized_states for s in rows]
        return states


