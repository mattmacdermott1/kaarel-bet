from kaarel_bet.analyse_results import token_prefix_prob


def test_token_prefix_prob_country_variants_sum():
    top5 = [
        {"token": "Un", "prob": 0.10},
        {"token": "United", "prob": 0.20},
        {"token": " united", "prob": 0.05},
        {"token": "King", "prob": 0.40},  # not a prefix of the full country
        {"token": "UK", "prob": 0.25},  # not a prefix of the full country
    ]
    # We expect only the first three to count toward "United Kingdom"
    p = token_prefix_prob(top5, "United Kingdom")
    assert abs(p - (0.10 + 0.20 + 0.05)) < 1e-9


def test_token_prefix_prob_capital_variants_sum():
    top5 = [
        {"token": " L", "prob": 0.10},  # leading space, should be stripped -> "l"
        {"token": "Lon", "prob": 0.15},  # prefix
        {"token": "london", "prob": 0.05},  # exact lowercased match
        {"token": "Paris", "prob": 0.40},  # not a prefix
        {"token": ",", "prob": 0.30},  # not a prefix
    ]
    p = token_prefix_prob(top5, "London")
    assert abs(p - (0.10 + 0.15 + 0.05)) < 1e-9


def test_token_prefix_prob_no_matches_is_zero():
    top5 = [
        {"token": "A", "prob": 0.5},
        {"token": "B", "prob": 0.5},
        {"token": " ", "prob": 0.0},
        {"token": "C", "prob": 0.0},
        {"token": "D", "prob": 0.0},
    ]
    p = token_prefix_prob(top5, "Zed")
    assert p == 0.0
