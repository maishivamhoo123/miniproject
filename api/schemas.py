from pydantic import BaseModel, Field

class AlloyInput(BaseModel):
    # All 19 Elemental compositions in atomic percent (must be between 0 and 100)
    Ag: float = Field(default=0.0, ge=0.0, le=100.0)
    Al: float = Field(default=0.0, ge=0.0, le=100.0)
    Au: float = Field(default=0.0, ge=0.0, le=100.0)
    Cd: float = Field(default=0.0, ge=0.0, le=100.0)
    Co: float = Field(default=0.0, ge=0.0, le=100.0)
    Cu: float = Field(default=0.0, ge=0.0, le=100.0)
    Fe: float = Field(default=0.0, ge=0.0, le=100.0)
    Hf: float = Field(default=0.0, ge=0.0, le=100.0)
    Mn: float = Field(default=0.0, ge=0.0, le=100.0)
    Nb: float = Field(default=0.0, ge=0.0, le=100.0)
    Ni: float = Field(default=0.0, ge=0.0, le=100.0)
    Pd: float = Field(default=0.0, ge=0.0, le=100.0)
    Pt: float = Field(default=0.0, ge=0.0, le=100.0)
    Ru: float = Field(default=0.0, ge=0.0, le=100.0)
    Si: float = Field(default=0.0, ge=0.0, le=100.0)
    Ta: float = Field(default=0.0, ge=0.0, le=100.0)
    Ti: float = Field(default=0.0, ge=0.0, le=100.0)
    Zn: float = Field(default=0.0, ge=0.0, le=100.0)
    Zr: float = Field(default=0.0, ge=0.0, le=100.0)
    
    # Processing & Physical Properties
    Cooling_Rate: float = Field(..., gt=0.0, description="Cooling rate in °C/min")
    Heating_Rate: float = Field(..., gt=0.0, description="Heating rate in °C/min")
    Density: float = Field(..., gt=0.0, description="Calculated Density in g/cm³")