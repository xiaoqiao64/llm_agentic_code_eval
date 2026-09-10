def final_price(price: float, tier: str) -> float:
    if tier == "gold":
        discount = price * 0.2
    elif tier == "silver":
        discount = price * 0.1
    else:
        discount = 0.0
    return price - discount
