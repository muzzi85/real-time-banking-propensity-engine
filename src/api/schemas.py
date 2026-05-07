from pydantic import BaseModel


class CustomerFeatures(BaseModel):

    age: int

    salary: float

    account_balance: float

    avg_card_spend: float

    tenure_months: int

    digital_engagement_score: float

    missed_payments: int

    is_sme_owner: int

    international_spend_ratio: float