import pytest
from backend.services.sla_forecasting_service import SLAForecastingService

def test_breach_probability_calculation():
    service = SLAForecastingService()
    prob = service.calculate_breach_probability(10, 0.9, 1.0)
    assert 0.0 <= prob <= 1.0
    assert prob > 0.5
