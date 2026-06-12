from pydantic import BaseModel, Field, model_validator


class Move(BaseModel):
    from_sq: int = Field(ge=0, le=31)
    to_sq: int = Field(ge=0, le=31)
    captured: list[int] = Field(default_factory=list)
    path: list[int] = Field(default_factory=list)

    def __repr__(self) -> str:
        cap = f" x{self.captured}" if self.captured else ""
        return f"Move({self.from_sq}->{self.to_sq}{cap})"


class BoardState(BaseModel):
    white_pieces: int = Field(ge=0, lt=2**32)
    black_pieces: int = Field(ge=0, lt=2**32)
    kings: int = Field(ge=0, lt=2**32)
    current_player: int = Field(ge=0, le=1)  # 1 = białe, 0 = czarne

    @model_validator(mode="after")
    def no_overlap(self) -> "BoardState":
        assert (self.white_pieces & self.black_pieces) == 0, (
            "Białe i czarne pionki nakładają się"
        )
        return self

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.white_pieces, self.black_pieces, self.kings, self.current_player)

    @classmethod
    def initial(cls) -> "BoardState":
        # Warcaby brazylijskie: czarne na polach 0-11, białe na 20-31
        black = (1 << 12) - 1          # bity 0..11
        white = ((1 << 12) - 1) << 20  # bity 20..31
        return cls(white_pieces=white, black_pieces=black, kings=0, current_player=1)
