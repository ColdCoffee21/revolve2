from __future__ import annotations

from typing import List, cast

import sqlalchemy
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.future import select
import sys

from .._serializer import Serializer


class FloatSerializer(Serializer[float]):
    @classmethod
    async def create_tables(cls, session: AsyncSession) -> None:
        await (await session.connection()).run_sync(DbBase.metadata.create_all)

    @classmethod
    def identifying_table(cls) -> str:
        return DbFloat.__tablename__

    @classmethod
    async def to_database(
        cls, session: AsyncSession, objects: List[dict]
    ) -> List[int]:

        # TODO: set attributes dynamically
        items = [DbFloat(
                         birth=f['birth'],
                         speed_y=f['speed_y'],
                         speed_x=f['speed_x'],
                         average_z=f['average_z'],
                         head_balance=f['head_balance'],
                         modules_count=f['modules_count'],
                         hinge_count=f['hinge_count'],
                         brick_count=f['brick_count'],
                         hinge_prop=f['hinge_prop'],
                         brick_prop=f['brick_prop'],
                         branching_count=f['branching_count'],
                         branching_prop=f['branching_prop'],
                         extremities=f['extremities'],
                         extensiveness=f['extensiveness'],
                         extremities_prop=f['extremities_prop'],
                         extensiveness_prop=f['extensiveness_prop'],
                         width=f['width'],
                         height=f['height'],
                         coverage=f['coverage'],
                         proportion=f['proportion'],
                         symmetry=f['symmetry'],
                         displacement=f['displacement'],
                         relative_speed_y=f['relative_speed_y'],
                         hinge_ratio=f['hinge_ratio'],
                         body_changes=f['body_changes'],
                         )
                 for f in objects]

        session.add_all(items)
        await session.flush()

        res = [
            i.id for i in items if i.id is not None
        ]  # is not None only there to silence mypy. can not actually be none because is marked not nullable.
        assert len(res) == len(objects)  # just to be sure now that we do the above

        return res

    @classmethod
    async def from_database(cls, session: AsyncSession, ids: List[int]) -> List[float]:
        items = (
            (await session.execute(select(DbFloat).filter(DbFloat.id.in_(ids))))
            .scalars()
            .all()
        )

        #measures_names = DbFloat.__table__.columns.keys()
        measures_genotypes = []
        for i in range(len(items)):
            measures = {}
            # TODO: do this dynamically using measures_names
            measures['birth'] = items[i].birth
            measures['speed_y'] = items[i].speed_y
            measures['speed_x'] = items[i].speed_x
            measures['average_z'] = items[i].average_z
            measures['head_balance'] = items[i].head_balance
            measures['modules_count'] = items[i].modules_count
            measures['hinge_count'] = items[i].hinge_count
            measures['brick_count'] = items[i].brick_count
            measures['hinge_prop'] = items[i].hinge_prop
            measures['brick_prop'] = items[i].brick_prop
            measures['branching_count'] = items[i].branching_count
            measures['branching_prop'] = items[i].branching_prop
            measures['extremities'] = items[i].extremities
            measures['extensiveness'] = items[i].extensiveness
            measures['extremities_prop'] = items[i].extremities_prop
            measures['extensiveness_prop'] = items[i].extensiveness_prop
            measures['width'] = items[i].width
            measures['height'] = items[i].height
            measures['coverage'] = items[i].coverage
            measures['proportion'] = items[i].proportion
            measures['symmetry'] = items[i].symmetry
            measures['relative_speed_y'] = items[i].relative_speed_y
            measures['displacement'] = items[i].displacement
            measures['hinge_ratio'] = items[i].hinge_ratio
            measures['body_changes'] = items[i].body_changes

            measures_genotypes.append(measures)

        return measures_genotypes


DbBase = declarative_base()


class DbFloat(DbBase):
    __tablename__ = "float"

    id = sqlalchemy.Column(
        sqlalchemy.Integer, nullable=False, primary_key=True, autoincrement=True
    )
    birth = sqlalchemy.Column(sqlalchemy.Float, nullable=True)
    speed_y = sqlalchemy.Column(sqlalchemy.Float, nullable=True)
    speed_x = sqlalchemy.Column(sqlalchemy.Float, nullable=True)
    relative_speed_y = sqlalchemy.Column(sqlalchemy.Float, nullable=True)
    displacement = sqlalchemy.Column(sqlalchemy.Float, nullable=True)
    average_z = sqlalchemy.Column(sqlalchemy.Float, nullable=True)
    head_balance = sqlalchemy.Column(sqlalchemy.Float, nullable=True)
    modules_count = sqlalchemy.Column(sqlalchemy.Float, nullable=True)
    hinge_count = sqlalchemy.Column(sqlalchemy.Float, nullable=True)
    brick_count = sqlalchemy.Column(sqlalchemy.Float, nullable=True)
    hinge_prop = sqlalchemy.Column(sqlalchemy.Float, nullable=True)
    brick_prop = sqlalchemy.Column(sqlalchemy.Float, nullable=True)
    branching_count = sqlalchemy.Column(sqlalchemy.Float, nullable=True)
    branching_prop = sqlalchemy.Column(sqlalchemy.Float, nullable=True)
    extremities = sqlalchemy.Column(sqlalchemy.Float, nullable=True)
    extensiveness = sqlalchemy.Column(sqlalchemy.Float, nullable=True)
    extremities_prop = sqlalchemy.Column(sqlalchemy.Float, nullable=True)
    extensiveness_prop = sqlalchemy.Column(sqlalchemy.Float, nullable=True)
    width = sqlalchemy.Column(sqlalchemy.Float, nullable=True)
    height = sqlalchemy.Column(sqlalchemy.Float, nullable=True)
    coverage = sqlalchemy.Column(sqlalchemy.Float, nullable=True)
    proportion = sqlalchemy.Column(sqlalchemy.Float, nullable=True)
    symmetry = sqlalchemy.Column(sqlalchemy.Float, nullable=True)
    hinge_ratio = sqlalchemy.Column(sqlalchemy.Float, nullable=True)
    body_changes = sqlalchemy.Column(sqlalchemy.Float, nullable=True)


