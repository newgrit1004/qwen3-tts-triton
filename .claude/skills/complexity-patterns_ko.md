# 복잡도 감소 패턴

Python 코드의 인지 복잡도를 줄이기 위한 참조 가이드입니다.

## 빠른 참조

| 패턴 | 복잡도 감소 | 사용 시점 |
|------|-------------|-----------|
| 조기 반환 | 레벨당 -2 ~ -5 | 깊은 중첩 |
| 함수 추출 | -3 ~ -10 | 긴 함수, 반복 로직 |
| 불린 헬퍼 | -1 ~ -3 | 복잡한 조건문 |
| 리스트 컴프리헨션 | -1 ~ -3 | 조건이 있는 단순 루프 |
| 가드 절 | -1 ~ -2 | 함수 시작 부분 검증 |
| 전략 패턴 | -5 ~ -15 | 큰 if/elif 체인 |

## 패턴 1: 조기 반환

중첩 조건문을 가드 절로 대체합니다.

### 이전 (복잡도: 5)
```python
def process_user(user):
    result = None
    if user is not None:
        if user.is_active:
            if user.has_permission:
                result = do_something(user)
    return result
```

### 이후 (복잡도: 2)
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

## 패턴 2: 함수 추출

큰 함수를 작고 집중된 함수로 분리합니다.

### 이전 (복잡도: 12)
```python
def process_order(order):
    # 검증 (복잡도: 4)
    if not order.items:
        raise ValueError("빈 주문")
    if order.total < 0:
        raise ValueError("잘못된 총액")
    for item in order.items:
        if item.quantity <= 0:
            raise ValueError("잘못된 수량")

    # 계산 (복잡도: 4)
    subtotal = 0
    for item in order.items:
        price = item.price * item.quantity
        if item.discount:
            price *= (1 - item.discount)
        subtotal += price

    # 세금과 배송 (복잡도: 4)
    tax = subtotal * 0.1
    shipping = 0
    if subtotal < 50:
        shipping = 10
    elif subtotal < 100:
        shipping = 5

    return subtotal + tax + shipping
```

### 이후 (각각 복잡도: 3)
```python
def process_order(order):
    validate_order(order)
    subtotal = calculate_subtotal(order)
    tax = calculate_tax(subtotal)
    shipping = calculate_shipping(subtotal)
    return subtotal + tax + shipping

def validate_order(order):
    if not order.items:
        raise ValueError("빈 주문")
    if order.total < 0:
        raise ValueError("잘못된 총액")
    for item in order.items:
        if item.quantity <= 0:
            raise ValueError("잘못된 수량")

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

## 패턴 3: 불린 헬퍼 함수

복잡한 불린 표현식을 추출합니다.

### 이전 (복잡도: 4)
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

### 이후 (복잡도: 1)
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

## 패턴 4: 리스트 컴프리헨션

단순 루프를 컴프리헨션으로 대체합니다.

### 이전 (복잡도: 4)
```python
def get_active_user_emails(users):
    result = []
    for user in users:
        if user.is_active:
            if user.email:
                result.append(user.email.lower())
    return result
```

### 이후 (복잡도: 1)
```python
def get_active_user_emails(users):
    return [
        user.email.lower()
        for user in users
        if user.is_active and user.email
    ]
```

## 패턴 5: 전략 패턴

큰 if/elif 체인을 디스패치 테이블로 대체합니다.

### 이전 (복잡도: 10)
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

### 이후 (복잡도: 2)
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
# ... 더 많은 핸들러
```

## 패턴 6: 검증을 위한 가드 절

검증을 함수 상단으로 이동합니다.

### 이전 (복잡도: 6)
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

### 이후 (복잡도: 2)
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

## 패턴 7: 조건문 대신 다형성

타입 검사 대신 클래스를 사용합니다.

### 이전 (복잡도: 8)
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
        raise ValueError(f"알 수 없는 타입: {item.type}")
```

### 이후 (클래스당 복잡도: 1-2)
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
        raise ValueError(f"알 수 없는 타입: {item.type}")
    return calculator.calculate(item)
```

## 복잡도 체크리스트

리팩토링 전:
- [ ] `uv run complexipy`로 현재 복잡도 측정
- [ ] 가장 높은 복잡도 함수 식별
- [ ] 적절한 패턴 선택

리팩토링 후:
- [ ] 복잡도 재측정
- [ ] 테스트 실행 (`make test`)
- [ ] 타입 검사 실행 (`make typecheck`)
- [ ] 동작 변경 없음 확인

## 일반적인 안티 패턴

| 안티 패턴 | 문제 | 해결책 |
|-----------|------|--------|
| 화살표 코드 | 깊은 중첩 | 조기 반환 |
| 만능 함수 | 너무 많은 책임 | 함수 추출 |
| 불린 파라미터 | 숨겨진 분기 | 별도 함수 |
| 중첩 삼항 연산자 | 읽기 어려움 | if 문 또는 헬퍼 |
| 모든 예외 잡기 | 숨겨진 복잡도 | 특정 핸들러 |
