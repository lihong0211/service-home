# 后端面试题精要（Python / Go）

> **姊妹文档：** 数据库（MySQL 详解）、PostgreSQL、MongoDB、Redis、消息队列（RabbitMQ / RocketMQ / Kafka）、网络与操作系统、分布式与系统设计等见 `[database-mq-interview.md](./database-mq-interview.md)`。

---

## 一、Python

### 1. Python 是解释型还是编译型？

Python 是**解释型**语言，源码 → 字节码（`.pyc`）→ CPython 虚拟机执行。  
并非直接机器码，但 Cython、PyPy 等可进一步编译优化。

---

### 2. GIL（全局解释器锁）

CPython 中的互斥锁，同一时刻只允许**一个线程**执行 Python 字节码。

- **影响：** CPU 密集型任务多线程无法真正并行
- **不影响：** I/O 密集型（I/O 等待时会释放 GIL）
- **绕过方案：** 多进程（`multiprocessing`）、C 扩展、`asyncio`

---

### 3. 深拷贝 vs 浅拷贝

```python
import copy

a = [[1, 2], [3, 4]]
b = a.copy()          # 浅拷贝：外层新对象，内层引用共享
c = copy.deepcopy(a)  # 深拷贝：完全独立

b[0].append(99)  # a[0] 也变了
c[0].append(99)  # a[0] 不变
```

---

### 4. 列表 vs 元组 vs 集合 vs 字典


|      | list | tuple         | set          | dict         |
| ---- | ---- | ------------- | ------------ | ------------ |
| 有序   | 是    | 是             | 否（3.7+ 插入有序） | 是（3.7+ 插入有序） |
| 可变   | 是    | **否**         | 是            | 是            |
| 允许重复 | 是    | 是             | **否**        | key 不重复      |
| 哈希   | 否    | **是**（可作 key） | 否            | 否            |


---

### 4.1 Python 数据类型

#### 核心分类

- **数值类型**：`int`、`float`、`complex`、`bool`
- **序列类型**：`str`、`list`、`tuple`、`range`
- **映射类型**：`dict`
- **集合类型**：`set`、`frozenset`
- **二进制类型**：`bytes`、`bytearray`、`memoryview`
- **空值类型**：`NoneType`（即 `None`）

#### 可变 vs 不可变（非常常考）

- **不可变**：`int`、`float`、`bool`、`str`、`tuple`、`frozenset`、`bytes`
- **可变**：`list`、`dict`、`set`、`bytearray`

```python
s = "abc"
print(id(s))
s += "d"         # 生成新对象
print(id(s))     # id 改变（不可变对象）

arr = [1, 2]
print(id(arr))
arr.append(3)    # 原地修改
print(id(arr))   # id 不变（可变对象）
```

#### 为什么有些类型能做 dict key？

- 作为 key 必须**可哈希（hashable）**
- 一般要求对象不可变，且 `__hash__` 稳定
- `tuple` 只有在其元素都可哈希时，才能做 key

```python
d = {}
d["name"] = "Tom"        # OK
d[(1, 2, 3)] = "tuple"   # OK
# d[[1, 2, 3]] = "list"  # TypeError: unhashable type: 'list'
```

#### 常见类型转换

```python
int("12")            # 12
float("3.14")        # 3.14
str(100)             # "100"
list("abc")          # ['a', 'b', 'c']
tuple([1, 2])        # (1, 2)
set([1, 1, 2, 3])    # {1, 2, 3}
dict([("a", 1)])     # {'a': 1}
```

#### 面试常见坑

1. `bool` 是 `int` 的子类：`isinstance(True, int) == True`
2. `is` 比较对象身份，`==` 比较值
3. `list * n` 复制嵌套列表会共享内层引用

```python
x = [[]] * 3
x[0].append(1)
print(x)   # [[1], [1], [1]]（都变了）
```

---

### 5. 装饰器（Decorator）

本质是**高阶函数**，在不修改原函数的前提下增强功能。

```python
import functools

def log(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"Done")
        return result
    return wrapper

@log
def greet(name): return f"Hello {name}"
```

**常见场景：** 日志、鉴权、缓存（`@lru_cache`）、重试、计时。

---

### 6. 生成器（Generator）

使用 `yield` 的函数，**惰性求值**，节省内存。

```python
def fib():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

gen = fib()
next(gen)  # 0
next(gen)  # 1
```

> 生成器表达式：`(x*x for x in range(100))` — 与列表推导式写法类似但是括号。

---

### 7. 迭代器协议

实现 `__iter__` 和 `__next__` 方法的对象。`for` 循环底层调用这两个方法，`StopIteration` 时停止。

```python
class CountDown:
    def __init__(self, start):
        self.n = start

    def __iter__(self):
        return self  # 迭代器对象通常返回自身

    def __next__(self):
        if self.n <= 0:
            raise StopIteration
        cur = self.n
        self.n -= 1
        return cur


for x in CountDown(3):
    print(x)   # 3 2 1
```

```python
it = iter(CountDown(2))  # 等价于调用 __iter__
print(next(it))          # 2（调用 __next__）
print(next(it))          # 1
# print(next(it))        # StopIteration
```

---

### 8. `*args` 和 `**kwargs`

```python
def func(*args, **kwargs):
    # args: 位置参数元组 → (1, 2, 3)
    # kwargs: 关键字参数字典 → {'a': 1, 'b': 2}
    pass

func(1, 2, 3, a=1, b=2)
```

---

### 9. 上下文管理器（with）

实现 `__enter__` 和 `__exit__`，保证资源释放（即使异常也执行 `__exit__`）。

```python
class Timer:
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args):
        print(f"Elapsed: {time.time() - self.start:.2f}s")

with Timer():
    time.sleep(1)
```

---

### 10. 闭包与作用域（LEGB）

Python 变量查找顺序：**L**ocal → **E**nclosing → **G**lobal → **B**uilt-in

```python
x = 'global'
def outer():
    x = 'enclosing'
    def inner():
        print(x)  # 'enclosing'（Enclosing 作用域）
    inner()
```

修改外层变量：`nonlocal x`（外层函数）/ `global x`（全局）

---

### 11. 类方法 / 静态方法 / 实例方法

```python
class MyClass:
    count = 0

    def instance_method(self):    # 第一个参数 self，访问实例
        return self

    @classmethod
    def class_method(cls):        # 第一个参数 cls，访问类本身
        return cls.count

    @staticmethod
    def static_method():          # 无隐式参数，与类无关的工具方法
        return "static"
```

---

### 12. `__new__` vs `__init__`

- `__new__`：创建实例（分配内存），返回实例对象
- `__init__`：初始化实例（设置属性），无返回值
- 单例模式通过重写 `__new__` 实现

```python
class Person:
    def __new__(cls, name):
        print("__new__ 调用：创建实例")
        instance = super().__new__(cls)
        return instance

    def __init__(self, name):
        print("__init__ 调用：初始化属性")
        self.name = name


p = Person("Tom")
print(p.name)
# 输出顺序：先 __new__，后 __init__
```

```python
class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)  # 只创建一次
        return cls._instance

    def __init__(self, value):
        self.value = value


a = Singleton(1)
b = Singleton(2)
print(a is b)   # True
print(a.value)  # 2（同一对象被再次初始化）
```

---

### 13. 多继承 & MRO（方法解析顺序）

Python 使用 **C3 线性化算法**确定 MRO，通过 `ClassName.__mro__` 查看顺序。

```python
class A: pass
class B(A): pass
class C(A): pass
class D(B, C): pass

D.__mro__  # D → B → C → A → object
```

---

### 14. 异常处理

```python
try:
    result = 1 / 0
except ZeroDivisionError as e:
    print(e)
except (TypeError, ValueError):
    pass
else:
    print("no exception")  # try 无异常时执行
finally:
    print("always runs")   # 无论如何都执行
```

---

### 15. asyncio / 协程

```python
import asyncio

async def fetch(url):
    await asyncio.sleep(1)  # 非阻塞等待
    return f"result: {url}"

async def main():
    results = await asyncio.gather(fetch("a"), fetch("b"))

asyncio.run(main())
```

- **协程**：单线程内的并发，适合 I/O 密集型（网络请求、数据库查询）
- **事件循环**：调度协程，`await` 处让出控制权

---

### 16. 常用内置函数

```python
map(fn, iterable)       # 映射
filter(fn, iterable)    # 过滤
zip(a, b)               # 打包
enumerate(iterable)     # 带索引遍历
sorted(iterable, key=fn, reverse=True)
any([False, True])      # True（有一个真）
all([True, True])       # True（全为真）
```

---

### 17. 内存管理

- **引用计数**：主要机制，引用数为 0 立即回收
- **循环垃圾收集器**：处理循环引用（标记-清除算法）
- **内存池**：小对象（≤256 字节）使用 pymalloc 内存池，避免频繁 malloc

---

### 18. Django vs Flask vs FastAPI


|     | Django        | Flask           | FastAPI                  |
| --- | ------------- | --------------- | ------------------------ |
| 类型  | 全栈重量级         | 微框架             | 现代异步框架                   |
| ORM | 内置 Django ORM | 无（用 SQLAlchemy） | 无（用 SQLAlchemy/Tortoise） |
| 异步  | 部分支持          | 有限支持            | ✅ 原生 async/await         |
| 性能  | 一般            | 中等              | 极高（接近 Go）                |
| 文档  | 手动            | 手动              | ✅ 自动（OpenAPI）            |
| 适用  | 快速搭建完整应用      | 灵活小项目           | 高性能 API、微服务              |


---

### 19. ORM 核心概念

SQLAlchemy 是 Python 最常用的 ORM，把数据库表映射成类、把行映射成对象，让我们用面向对象方式操作数据库。  
核心是 `Session` 作为工作单元统一管理增删改查和事务提交，常用查询是链式写法：`query().filter().order_by().limit().all()`。

```python
# SQLAlchemy 示例
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String(50))

# 查询
session.query(User).filter(User.name == 'Alice').first()

# N+1 问题：查询列表后逐条查关联数据
# 解决：使用 joinedload / selectinload 预加载
```

---

### 20. RESTful API 设计规范

```
GET    /users          # 获取列表
GET    /users/{id}     # 获取单个
POST   /users          # 创建
PUT    /users/{id}     # 全量更新
PATCH  /users/{id}     # 部分更新
DELETE /users/{id}     # 删除

状态码：200、201、204、400、401、403、404、500
版本控制：/api/v1/users
```

---

### 21. JWT 认证

**JSON Web Token**：把「身份与权限声明」编码成一串可验证的字符串，常用于 **API 无状态认证**（也可做信息交换，但认证场景最常见）。

#### 结构：`Header.Payload.Signature`（两段 Base64URL + 签名）

三部分用 **`.`** 连接；**Header** 与 **Payload** 是 **Base64URL** 编码的 JSON（**不是加密**，任何人都能解码看明文，**敏感信息不要放 Payload**）。

```
eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c
```

| 部分 | 内容 | 作用 |
|---|---|---|
| **Header** | 如 `{"alg":"HS256","typ":"JWT"}` | 声明签名算法与类型 |
| **Payload** | 一组 **Claims**（声明）| 承载用户 id、角色、过期时间等 |
| **Signature** | 对 `Base64Url(Header) + "." + Base64Url(Payload)` 做签名 | 防篡改；校验通过才信任 Payload |

**签名算法（常见）：**

- **HS256** 等对称算法：服务端用**同一密钥**签发与校验，实现简单，密钥泄露则全线危险。
- **RS256** 等非对称算法：**私钥签发、公钥校验**，适合多服务/网关只持公钥验签，私钥集中在认证服务。

#### 标准 Claims（RFC 7519，面试常问）

| Claim | 含义 |
|---|---|
| `iss` | 签发者 |
| `sub` | 主体（常用用户 id）|
| `aud` | 受众（期望消费的系统）|
| `exp` | 过期时间（Unix 时间戳）|
| `nbf` | 生效时间 |
| `iat` | 签发时间 |
| `jti` | 令牌唯一 id（可做一次性或黑名单键）|

自定义业务字段如 `role`、`permissions` 也可放 Payload，但**体积越大，每次请求头越大**。

#### 典型认证流程

1. 用户登录（账号密码 / OAuth 等），服务端校验通过后**签发 JWT**（设较短 `exp`，如 15 分钟～2 小时）。
2. 客户端保存（**HttpOnly Cookie** 可降低 XSS 盗 token 风险；若放 **localStorage** 则须严防 XSS）。
3. 后续请求带 `Authorization: Bearer <token>`（或 Cookie），**网关 / 业务服务**用密钥或公钥**验签**并解析 `exp`、`sub` 等。
4. **无需**为每次请求查库会话表（**无状态**）；需要权限时再查库或查缓存亦可。

#### 优点

- **无状态**：水平扩展服务时不依赖集中 Session 存储。
- **跨服务**：各服务共密钥或共公钥即可验签（注意 `aud`、时钟同步）。
- **适合移动端 / SPA**：天然适合 Bearer 传递。

#### 缺点与风险

- **默认无法「服务端主动作废」**：在 `exp` 之前 token 一直有效（除非维护**黑名单**、或改密钥全员失效）。
- **泄露即冒用**：HTTPS 传输；避免把 JWT 打到日志、URL 查询串。
- **Payload 明文**：勿放密码、支付密钥等。

#### 工程上常见补强

- **Access Token（短效）+ Refresh Token（长效、可轮转、可存库或 Redis）**：降低被盗窗口，刷新时可吊销 refresh。
- **黑名单 / 版本号**：登出或封号时把 `jti` 或 `(user_id, token_version)` 记入 Redis，校验时拒绝。
- **权限变更**：Payload 里的角色若长期不变，改权限后需等新 token 或配合服务端二次校验。

#### 与 Session 对比（一句话）

| | Session（Cookie + 服务端存储）| JWT |
|---|---|---|
| 状态 | 有状态，依赖存储 | 验签即可，默认无状态 |
| 吊销 | 删 session 即失效 | 需额外机制（短 exp、黑名单、refresh）|
| 体积 | Cookie 里常只存 session id | 整包 claims 在客户端，可能较大 |

---

**速记公式（HS256）：**

```
Signature = HMACSHA256( base64url(Header) + "." + base64url(Payload), secret )
```

---

## 二、Go

### 1. Go 是编译型还是解释型？

Go 是**编译型**语言，源码直接编译为**机器码**（静态链接），单二进制部署，无运行时依赖。

---

### 2. goroutine 与 GMP 模型

- **goroutine**：用户态轻量级线程，创建与切换成本低，可成千上万
- **GMP**：**G**oroutine、**M**achine（OS 线程）、**P**rocessor（调度上下文）
  - P 数量 ≈ GOMAXPROCS，M 与 P 绑定执行 G
  - 当 G 阻塞时，M 可解绑 P，让 P 去执行其他 G，避免线程堵死

---

### 3. channel 与 select

```go
ch := make(chan int, 10)   // 带缓冲
ch <- 1                    // 写
v := <-ch                  // 读
close(ch)                   // 关闭后不可写，可读零值

select {
case v := <-ch:
    fmt.Println(v)
case ch <- 1:
default:
    // 非阻塞
}
```

**注意：** 向已 close 的 channel 写会 panic；从已 close 的 channel 读会拿到零值并 ok=false。

---

### 4. defer 执行顺序

多个 `defer` 按**后进先出（LIFO）**执行。

```go
defer fmt.Println("1")
defer fmt.Println("2")
// 输出：2 → 1
```

defer 在**函数返回前**执行，可修改命名返回值。

---

### 5. 值类型 vs 引用类型

- **值类型**：int、float、bool、string、array、struct — 赋值/传参是拷贝
- **引用类型**：slice、map、channel — 底层共享数据，拷贝的是“描述符”

```go
s := []int{1, 2}
s2 := s
s2[0] = 99  // s[0] 也变成 99
```

---

### 6. slice 与 map 的底层

- **slice**：`ptr + len + cap`，底层数组可共享；append 超 cap 会重新分配并拷贝
- **map**：哈希表实现，**非线程安全**，并发写需用 `sync.Mutex` 或 `sync.Map`

---

### 7. 接口（interface）与类型断言

```go
var i interface{} = 42
v, ok := i.(int)    // 类型断言，ok 表示是否成功
```

**空接口** `interface{}` 等价于 `any`（Go 1.18+），可承载任意类型。

---

### 8. 并发安全：sync.Mutex / sync.RWMutex

- **Mutex**：互斥锁，Lock/Unlock
- **RWMutex**：读多写少时，RLock/RUnlock 允许多读，Lock/Unlock 独占写

**注意：** 锁不要复制，传递时用指针。

---

### 9. context 包

用于**取消、超时、传值**，在 goroutine 树中传递。

```go
ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
defer cancel()

select {
case <-ctx.Done():
    fmt.Println(ctx.Err())  // context.DeadlineExceeded
}
```

---

### 10. 内存与 GC

- **逃逸分析**：变量若在堆上分配会“逃逸”，编译器通过逃逸分析尽量把对象放在栈上
- **GC**：三色标记 + 写屏障，并发标记与清扫，STW 很短

---

### 11. 常见标准库与生态

- **net/http**：HTTP 服务与客户端
- **encoding/json**：JSON 序列化（struct tag：`json:"name"`）
- **database/sql**：统一数据库接口，需驱动（如 `go-sql-driver/mysql`）
- **gin / echo / fiber**：常用 Web 框架

---

### 12. 错误处理

Go 无 try-catch，通过**返回值**传递错误。

```go
if err != nil {
    return fmt.Errorf("open file: %w", err)  // %w 包装，errors.Is/As 可解包
}
```

---

*Python / Go 部分持续更新；基础设施与中间件见 `database-mq-interview.md`。*