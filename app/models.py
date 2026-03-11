from pydantic import BaseModel, Field


class MovieRecommendation(BaseModel):
    title: str = Field(description="The movie title mentioned in the review or most probable movie title if not mentioned")
    reason: str = Field(description="Why this movie matches the user's preferences")
    genre: str = Field(description="The genre(s) of the movie")


class MovieRecommendations(BaseModel):
    recommendations: list[MovieRecommendation] = Field(
        description="List of recommended movies"
    )
    summary: str = Field(
        description="Brief overall summary of the recommendations"
    )
