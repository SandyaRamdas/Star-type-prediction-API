from pydantic import BaseModel

class StarInput(BaseModel):
    temperature : int
    luminosity : float
    radius : float
    absolute_magnitude : float