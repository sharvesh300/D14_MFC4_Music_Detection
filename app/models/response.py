from dataclasses import dataclass, field


@dataclass
class MatchResult:
    song_id: int
    song_name: str
    confidence: float


@dataclass
class MatchResponse:
    query_path: str
    n_hashes: int
    matched: bool
    results: list[MatchResult] = field(default_factory=list)
