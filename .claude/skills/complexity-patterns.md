# Complexity Reduction Patterns

Reference guide for reducing cognitive complexity in Python code.

## Quick Reference

| Pattern | Complexity Reduction | When to Use |
|---------|---------------------|-------------|
| Early Return | -2 to -5 per level | Deep nesting |
| Extract Function | -3 to -10 | Long functions, repeated logic |
| Boolean Helper | -1 to -3 | Complex conditionals |
| List Comprehension | -1 to -3 | Simple loops with conditions |
| Guard Clause | -1 to -2 | Validation at function start |
| Strategy Pattern | -5 to -15 | Large if/elif chains |

## Pattern 1: Early Return

Replace nested conditionals with guard clauses.

### Before (Complexity: 5)
```python
def process_user(user):
    result = None
    if user is not None:
        if user.is_active:
            if user.has_permission:
                result = do_something(user)
    return result
```

### After (Complexity: 2)
```python
def process_user(user):
    if user is None:
        return None
    if not user.is_active:
        return None
    if not user.has_permission:
        return None
    return do_something(user)
```

## Pattern 2: Extract Function

Break large functions into smaller, focused ones.

### Before (Complexity: 12)
```python
def process_order(order):
    # Validation (complexity: 4)
    if not order.items:
        raise ValueError("Empty order")
    if order.total < 0:
        raise ValueError("Invalid total")
    for item in order.items:
        if item.quantity <= 0:
            raise ValueError("Invalid quantity")

    # Calculation (complexity: 4)
    subtotal = 0
    for item in order.items:
        price = item.price * item.quantity
        if item.discount:
            price *= (1 - item.discount)
        subtotal += price

    # Tax and shipping (complexity: 4)
    tax = subtotal * 0.1
    shipping = 0
    if subtotal < 50:
        shipping = 10
    elif subtotal < 100:
        shipping = 5

    return subtotal + tax + shipping
```

### After (Complexity: 3 each)
```python
def process_order(order):
    validate_order(order)
    subtotal = calculate_subtotal(order)
    tax = calculate_tax(subtotal)
    shipping = calculate_shipping(subtotal)
    return subtotal + tax + shipping

def validate_order(order):
    if not order.items:
        raise ValueError("Empty order")
    if order.total < 0:
        raise ValueError("Invalid total")
    for item in order.items:
        if item.quantity <= 0:
            raise ValueError("Invalid quantity")

def calculate_subtotal(order):
    return sum(
        item.price * item.quantity * (1 - (item.discount or 0))
        for item in order.items
    )

def calculate_tax(subtotal):
    return subtotal * 0.1

def calculate_shipping(subtotal):
    if subtotal < 50:
        return 10
    if subtotal < 100:
        return 5
    return 0
```

## Pattern 3: Boolean Helper Functions

Extract complex boolean expressions.

### Before (Complexity: 4)
```python
def can_access_resource(user, resource):
    if (user.is_admin or
        (user.is_active and user.has_subscription and
         resource.is_public) or
        (user.id in resource.allowed_users and
         not resource.is_locked)):
        return True
    return False
```

### After (Complexity: 1)
```python
def can_access_resource(user, resource):
    return (
        is_admin(user) or
        has_public_access(user, resource) or
        has_explicit_access(user, resource)
    )

def is_admin(user):
    return user.is_admin

def has_public_access(user, resource):
    return user.is_active and user.has_subscription and resource.is_public

def has_explicit_access(user, resource):
    return user.id in resource.allowed_users and not resource.is_locked
```

## Pattern 4: List Comprehension

Replace simple loops with comprehensions.

### Before (Complexity: 4)
```python
def get_active_user_emails(users):
    result = []
    for user in users:
        if user.is_active:
            if user.email:
                result.append(user.email.lower())
    return result
```

### After (Complexity: 1)
```python
def get_active_user_emails(users):
    return [
        user.email.lower()
        for user in users
        if user.is_active and user.email
    ]
```

## Pattern 5: Strategy Pattern

Replace large if/elif chains with dispatch tables.

### Before (Complexity: 10)
```python
def handle_event(event):
    if event.type == "user_created":
        send_welcome_email(event.data)
        create_default_settings(event.data)
    elif event.type == "user_updated":
        sync_user_data(event.data)
        invalidate_cache(event.data)
    elif event.type == "user_deleted":
        archive_user_data(event.data)
        cleanup_resources(event.data)
    elif event.type == "order_placed":
        process_payment(event.data)
        send_confirmation(event.data)
    else:
        log_unknown_event(event)
```

### After (Complexity: 2)
```python
EVENT_HANDLERS = {
    "user_created": handle_user_created,
    "user_updated": handle_user_updated,
    "user_deleted": handle_user_deleted,
    "order_placed": handle_order_placed,
}

def handle_event(event):
    handler = EVENT_HANDLERS.get(event.type)
    if handler:
        handler(event.data)
    else:
        log_unknown_event(event)

def handle_user_created(data):
    send_welcome_email(data)
    create_default_settings(data)

def handle_user_updated(data):
    sync_user_data(data)
    invalidate_cache(data)
# ... more handlers
```

## Pattern 6: Guard Clause for Validation

Move validation to the top of functions.

### Before (Complexity: 6)
```python
def calculate_discount(order, coupon):
    discount = 0
    if order.total > 0:
        if coupon is not None:
            if coupon.is_valid:
                if coupon.min_purchase <= order.total:
                    if coupon.type == "percentage":
                        discount = order.total * coupon.value / 100
                    else:
                        discount = coupon.value
    return discount
```

### After (Complexity: 2)
```python
def calculate_discount(order, coupon):
    if order.total <= 0:
        return 0
    if coupon is None or not coupon.is_valid:
        return 0
    if coupon.min_purchase > order.total:
        return 0

    if coupon.type == "percentage":
        return order.total * coupon.value / 100
    return coupon.value
```

## Pattern 7: Polymorphism Over Conditionals

Use classes instead of type checking.

### Before (Complexity: 8)
```python
def calculate_shipping(item):
    if item.type == "physical":
        if item.weight < 1:
            return 5
        elif item.weight < 5:
            return 10
        else:
            return 20
    elif item.type == "digital":
        return 0
    elif item.type == "subscription":
        return 0
    else:
        raise ValueError(f"Unknown type: {item.type}")
```

### After (Complexity: 1-2 per class)
```python
from abc import ABC, abstractmethod

class ShippingCalculator(ABC):
    @abstractmethod
    def calculate(self, item) -> float:
        pass

class PhysicalShipping(ShippingCalculator):
    def calculate(self, item) -> float:
        if item.weight < 1:
            return 5
        if item.weight < 5:
            return 10
        return 20

class DigitalShipping(ShippingCalculator):
    def calculate(self, item) -> float:
        return 0

SHIPPING_CALCULATORS = {
    "physical": PhysicalShipping(),
    "digital": DigitalShipping(),
    "subscription": DigitalShipping(),
}

def calculate_shipping(item):
    calculator = SHIPPING_CALCULATORS.get(item.type)
    if not calculator:
        raise ValueError(f"Unknown type: {item.type}")
    return calculator.calculate(item)
```

## Complexity Checklist

Before refactoring:
- [ ] Measure current complexity with `uv run complexipy`
- [ ] Identify highest complexity functions
- [ ] Choose appropriate pattern(s)

After refactoring:
- [ ] Re-measure complexity
- [ ] Run tests (`make test`)
- [ ] Run type check (`make typecheck`)
- [ ] Verify behavior unchanged

## Common Anti-Patterns

| Anti-Pattern | Issue | Solution |
|--------------|-------|----------|
| Arrow code | Deep nesting | Early returns |
| God function | Too many responsibilities | Extract functions |
| Boolean parameters | Hidden branching | Separate functions |
| Nested ternaries | Hard to read | If statements or helpers |
| Catch-all exception | Hidden complexity | Specific handlers |
