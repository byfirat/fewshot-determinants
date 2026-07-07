from dataclasses import asdict, dataclass
from datetime import datetime


@dataclass
class RunMetadata:
    run_id: str
    experiment_name: str
    created_at: str

    @classmethod
    def create(cls, experiment_name: str) -> "RunMetadata":
        now = datetime.now()
        run_id = now.strftime("%Y%m%d_%H%M%S") + f"_{experiment_name}"
        return cls(run_id=run_id, experiment_name=experiment_name, created_at=now.isoformat())

    def to_dict(self):
        return asdict(self)
