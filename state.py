from pydantic import BaseModel, Field
from typing import Optional

class LoanState(BaseModel):
    user_input: str
    salary: Optional[int] = Field(default=None, description="Monthly in-hand salary")
    credit_score: Optional[int] = Field(default=None, description="CIBIL/Credit Score")
    loan_amount: Optional[int] = Field(default=None, description="Requested loan amount")
    tenure_years: Optional[int] = Field(default=None, description="Requested loan tenure in years")
    existing_emi: Optional[int] = Field(default=0, description="Sum of existing monthly EMIs")

    retrieved_rules: Optional[str] = None
    eligibility_status: Optional[str] = None
    approval_score: Optional[float] = None
    suggested_plan: Optional[str] = None
    final_reply: Optional[str] = None
