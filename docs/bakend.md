# 后端面试题精要

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

| | list | tuple | set | dict |
|---|---|---|---|---|
| 有序 | 是 | 是 | 否（3.7+ 插入有序）| 是（3.7+ 插入有序）|
| 可变 | 是 | **否** | 是 | 是 |
| 允许重复 | 是 | 是 | **否** | key 不重复 |
| 哈希 | 否 | **是**（可作 key）| 否 | 否 |

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

## 三、数据库（MySQL）

### 1. 事务 ACID

| 特性 | 说明 |
|---|---|
| **A** tomicity 原子性 | 事务要么全成功，要么全回滚 |
| **C** onsistency 一致性 | 事务前后数据符合业务规则 |
| **I** solation 隔离性 | 并发事务互不干扰 |
| **D** urability 持久性 | 提交后数据永久保存（redo log）|

---

### 2. 事务隔离级别

| 级别 | 脏读 | 不可重复读 | 幻读 |
|---|---|---|---|
| Read Uncommitted | ✅ 有 | ✅ 有 | ✅ 有 |
| Read Committed | ❌ 无 | ✅ 有 | ✅ 有 |
| **Repeatable Read**（MySQL 默认）| ❌ 无 | ❌ 无 | ⚠️ 部分（MVCC 解决快照读）|
| Serializable | ❌ 无 | ❌ 无 | ❌ 无 |

- **脏读：** 读到其他事务未提交的数据
- **不可重复读：** 同一事务内两次读同一行，数据不同（被其他事务修改）
- **幻读：** 同一事务内两次查询，行数不同（被其他事务插入/删除）

---

### 3. 索引类型与原理

**结构：** InnoDB 使用 **B+ 树**索引

| 类型 | 说明 |
|---|---|
| 主键索引（聚簇）| 叶子节点存储完整行数据 |
| 普通索引（非聚簇）| 叶子节点存储主键值，需**回表**查询 |
| 唯一索引 | 保证字段值唯一 |
| 复合索引 | 多列联合索引，遵循**最左前缀原则** |
| 全文索引 | 文本搜索（倒排索引）|

**B+ 树 vs B 树：**
- B+ 树所有数据在叶子节点，叶子节点有链表连接 → 范围查询高效
- 非叶子节点只存 key，单页能存更多，树更矮（减少 I/O）

---

### 4. 最左前缀原则

复合索引 `(a, b, c)`：

```sql
WHERE a = 1            -- ✅ 使用索引
WHERE a = 1 AND b = 2  -- ✅ 使用索引
WHERE b = 2            -- ❌ 不使用（跳过 a）
WHERE a = 1 AND c = 3  -- ✅ 部分使用（a 索引，c 无法用）
WHERE a > 1 AND b = 2  -- ✅ a 使用，b 范围查询后失效
```

---

### 5. EXPLAIN 字段解读

```sql
EXPLAIN SELECT * FROM orders WHERE user_id = 1;
```

| 字段 | 关注点 |
|---|---|
| `type` | 性能从好到差：system > const > eq_ref > ref > range > **index > ALL** |
| `key` | 实际使用的索引（NULL 表示未用索引）|
| `rows` | 预估扫描行数（越小越好）|
| `Extra` | `Using index`（覆盖索引好）、`Using filesort`（需优化）|

---

### 6. 索引失效场景

```sql
-- 1. 对索引列使用函数
WHERE YEAR(create_time) = 2024       -- ❌

-- 2. 隐式类型转换
WHERE phone = 13800138000            -- phone 是 varchar，❌

-- 3. 前导模糊查询
WHERE name LIKE '%张'                 -- ❌（'张%' 可以）

-- 4. OR 连接非索引列
WHERE id = 1 OR name = 'John'        -- name 无索引则全表扫描

-- 5. NOT IN / NOT EXISTS（部分情况）

-- 6. 范围查询后的字段（复合索引）
```

---

### 7. 锁机制

| 锁类型 | 说明 |
|---|---|
| 共享锁（S）| 读锁，多个事务可同时持有 |
| 排他锁（X）| 写锁，独占，其他事务不能读写 |
| 意向锁 | 表级，标记某行已被行锁，避免表锁与行锁冲突检查 |
| 行锁 | 锁定特定行（InnoDB 支持）|
| 间隙锁（Gap Lock）| 锁定索引间隙，防止幻读插入 |
| 临键锁（Next-Key Lock）| 行锁 + 间隙锁，RR 级别默认 |

**死锁：** 两个事务互相等待对方持有的锁。MySQL 自动检测并回滚代价小的事务。

---

### 8. MVCC（多版本并发控制）

InnoDB 在 Repeatable Read 下通过 MVCC 实现无锁读：
- 每行数据有隐藏字段：`trx_id`（最近修改事务 ID）、`roll_pointer`（undo log 指针）
- 读操作生成**一致性视图（Read View）**，根据事务 ID 判断版本可见性
- **快照读**（普通 SELECT）：不加锁，读历史版本
- **当前读**（SELECT FOR UPDATE / INSERT / UPDATE / DELETE）：加锁，读最新版本

---

### 9. redo log vs undo log vs binlog

| | redo log | undo log | binlog |
|---|---|---|---|
| 所属层 | InnoDB 引擎层 | InnoDB 引擎层 | MySQL Server 层 |
| 作用 | 崩溃恢复（持久性）| 事务回滚 + MVCC | 主从同步、数据恢复 |
| 格式 | 物理日志（页变更）| 逻辑日志（反操作）| 逻辑日志（statement/row/mixed）|
| 写入时机 | 事务执行中（WAL）| 事务开始时 | 事务提交时 |

**WAL（Write-Ahead Logging）：** 先写日志再写磁盘，减少随机 I/O。

---

### 10. 分库分表

**水平分表：** 同一结构，数据按规则拆分到多张表（如按 user_id 取模）  
**垂直分表：** 按字段拆分，常用字段和大字段分离（冷热分离）  
**分库：** 将不同业务的表放到不同数据库（垂直分库）或数据水平拆分到多个库

**分片键选择原则：** 数据均匀分布、查询尽量落在单库、避免跨库事务

---

### 11. 慢查询优化思路

1. 开启慢查询日志，找到慢 SQL
2. `EXPLAIN` 分析执行计划
3. 检查是否命中索引（type、key 字段）
4. 优化：加索引、改写 SQL、覆盖索引减少回表
5. 大表考虑分页优化（`WHERE id > last_id LIMIT 10` 代替 `OFFSET`）
6. 必要时读写分离、缓存热点数据

---

### 12. InnoDB vs MyISAM

| | InnoDB | MyISAM |
|---|---|---|
| 事务 | ✅ 支持 | ❌ |
| 外键 | ✅ 支持 | ❌ |
| 行锁 | ✅ | ❌（表锁）|
| 崩溃恢复 | ✅（redo log）| ❌ |
| 全文索引 | ✅（5.6+）| ✅ |
| 适用场景 | OLTP（事务读写）| 读多写少（已基本淘汰）|

---

---

## 四、Redis

### 1. Redis 数据类型

| 类型 | 底层实现 | 场景 |
|---|---|---|
| String | SDS（动态字符串）| 缓存、计数器、分布式锁 |
| Hash | 压缩列表 / 哈希表 | 对象存储（用户信息）|
| List | 压缩列表 / 双向链表 | 消息队列、最新列表 |
| Set | 哈希表 / 整数集合 | 去重、标签、共同好友 |
| ZSet（Sorted Set）| 压缩列表 / 跳表 | 排行榜、延迟队列 |
| Stream | - | 消息队列（Pub/Sub 升级版）|
| BitMap | - | 签到、在线状态 |
| HyperLogLog | - | 大数据基数统计（有误差）|

---

### 2. Redis 为什么快？

1. **纯内存操作**：数据在内存，读写纳秒级
2. **单线程模型**（命令处理）：无锁竞争，无上下文切换
3. **I/O 多路复用**：`epoll` 事件驱动，处理大量并发连接
4. **高效数据结构**：SDS、跳表、压缩列表等针对场景优化

> Redis 6.0+ 引入多线程处理网络 I/O，命令执行仍单线程。

---

### 3. 持久化：RDB vs AOF

| | RDB | AOF |
|---|---|---|
| 方式 | 全量快照（fork 子进程）| 追加写命令日志 |
| 数据安全 | 可能丢失最后一次快照后的数据 | 最多丢失 1 秒数据（fsync 策略）|
| 文件大小 | 小（二进制压缩）| 大（命令文本，可 rewrite 压缩）|
| 恢复速度 | 快 | 慢（回放命令）|
| 生产建议 | 同时开启 | 同时开启，AOF 优先恢复 |

---

### 4. 缓存三问：穿透、击穿、雪崩

**缓存穿透**（查询不存在的数据）
- 原因：恶意请求不存在的 key，每次都打到 DB
- 解决：缓存空值（短 TTL）、**布隆过滤器**（前置过滤）

**缓存击穿**（热点 key 过期）
- 原因：热点 key 突然过期，大量请求打到 DB
- 解决：**互斥锁**（只允许一个请求重建缓存）、逻辑过期（不设 TTL，异步更新）

**缓存雪崩**（大量 key 同时过期 / Redis 宕机）
- 原因：批量 key 相同时间过期，或 Redis 实例崩溃
- 解决：TTL **随机化**、Redis **集群/哨兵**高可用、熔断降级

---

### 5. 分布式锁

```bash
# 原子操作：加锁
SET lock_key unique_value NX PX 30000
# NX: 不存在才设置（互斥）
# PX 30000: 30秒过期（防死锁）

# 释放锁（Lua 脚本保证原子性）
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
end
```

**Redlock 算法：** 多个独立 Redis 实例，向超半数节点加锁才算成功，防单点故障。

---

### 6. 淘汰策略（内存满时）

| 策略 | 说明 |
|---|---|
| `noeviction` | 不淘汰，写操作报错（默认）|
| `allkeys-lru` | 所有 key 中 LRU 淘汰（常用）|
| `volatile-lru` | 有 TTL 的 key 中 LRU 淘汰 |
| `allkeys-lfu` | 所有 key 中 LFU 淘汰（Redis 4.0+）|
| `volatile-ttl` | TTL 最短的先淘汰 |
| `allkeys-random` | 随机淘汰 |

---

### 7. 主从复制

1. **全量复制**：从节点首次连接，主节点 fork 生成 RDB 快照发送
2. **增量复制**：断线重连后，主节点通过 `repl_backlog`（环形缓冲区）补发缺失命令
3. **复制延迟**：异步复制，存在数据延迟（可用 `WAIT` 命令半同步）

---

### 8. 哨兵（Sentinel）vs 集群（Cluster）

| | Sentinel | Cluster |
|---|---|---|
| 目的 | 高可用（自动故障转移）| 高可用 + 水平扩展 |
| 数据分片 | 否（全量在主节点）| 是（16384 槽位哈希分片）|
| 节点数 | 1 主多从 + 3+ 哨兵 | 3+ 主节点，每主带从节点 |
| 客户端 | 连哨兵获取主节点地址 | 直连集群，支持重定向 |
| 适合场景 | 数据量不大、读写分离 | 大数据量、高并发 |

---

### 9. 跳表（SkipList）

ZSet 的底层结构，有序链表 + 多层索引，查询/插入/删除 O(log N)。  
优于红黑树：实现简单，范围查询友好；优于 B+ 树：不需要节点合并分裂。

---

### 10. Pipeline & Lua 脚本

**Pipeline：** 批量发送命令，减少 RTT（网络往返）次数，提升吞吐量。非原子。

**Lua 脚本：** Redis 保证脚本原子执行（执行期间不处理其他命令），适合复合操作。

```bash
EVAL "return redis.call('set', KEYS[1], ARGV[1])" 1 mykey myvalue
```

---

---

## 五、网络 & 操作系统

### 1. TCP 三次握手 / 四次挥手

**三次握手（建立连接）：**
```
客户端 → SYN(seq=x)            → 服务端  [SYN_SENT]
客户端 ← SYN+ACK(seq=y,ack=x+1)← 服务端  [SYN_RCVD]
客户端 → ACK(ack=y+1)          → 服务端  [ESTABLISHED]
```

**四次挥手（关闭连接）：**
```
客户端 → FIN → 服务端   [FIN_WAIT_1]
客户端 ← ACK ← 服务端   [CLOSE_WAIT] 服务端还可发数据
客户端 ← FIN ← 服务端   [LAST_ACK]
客户端 → ACK → 服务端   [TIME_WAIT → CLOSED]（等 2MSL）
```

> 为什么三次握手？确认双方收发能力正常；两次不够，服务端无法确认客户端能收。  
> 为什么四次挥手？TCP 半关闭，服务端收到 FIN 后还可能有数据要发。

---

### 2. TCP vs UDP

| | TCP | UDP |
|---|---|---|
| 连接 | 面向连接 | 无连接 |
| 可靠性 | 可靠（重传、确认）| 不可靠 |
| 顺序 | 保证顺序 | 不保证 |
| 速度 | 慢（握手、拥塞控制）| 快 |
| 场景 | HTTP、FTP、SSH | 视频流、DNS、游戏 |

---

### 3. HTTP/1.1 vs HTTP/2 vs HTTP/3

| | HTTP/1.1 | HTTP/2 | HTTP/3 |
|---|---|---|---|
| 传输 | 文本 | 二进制帧 | QUIC（UDP）|
| 多路复用 | 有队头阻塞 | ✅（帧级别）| ✅（流级别，彻底解决）|
| 头部压缩 | ❌ | ✅（HPACK）| ✅（QPACK）|
| 服务器推送 | ❌ | ✅ | ✅ |

---

### 4. 进程 vs 线程 vs 协程

| | 进程 | 线程 | 协程 |
|---|---|---|---|
| 内存空间 | 独立 | 共享进程内存 | 共享线程内存 |
| 切换开销 | 大 | 中 | 极小 |
| 并发 | 是 | 是 | 单线程内并发 |
| 通信 | IPC（管道、信号等）| 共享内存 | 函数调用 |
| 崩溃影响 | 不影响其他进程 | 影响同进程所有线程 | — |

---

### 5. 同步 / 异步 / 阻塞 / 非阻塞

- **阻塞 vs 非阻塞：** 等待结果时是否占用 CPU（调用方的状态）
- **同步 vs 异步：** 调用方是否等待结果返回再继续（消息通知机制）

| 组合 | 说明 | 示例 |
|---|---|---|
| 同步阻塞 | 最常见，等待完成再继续 | 普通 read() |
| 同步非阻塞 | 轮询检查，不等待 | 非阻塞 socket |
| 异步非阻塞 | 注册回调，完成后通知 | epoll + callback |

---

### 6. 负载均衡算法

| 算法 | 说明 | 适用 |
|---|---|---|
| 轮询（Round Robin）| 依次分配 | 服务器性能相同 |
| 加权轮询 | 按权重分配 | 性能不同的服务器 |
| IP Hash | 同一 IP 到同一服务器 | 需要会话保持 |
| 最少连接 | 分配给连接数最少的 | 长连接场景 |
| 随机 | 随机选择 | 简单场景 |

---

---

## 六、系统设计 & 架构

### 1. CAP 定理

分布式系统只能同时满足以下两项：
- **C**onsistency（一致性）：所有节点同时看到相同数据
- **A**vailability（可用性）：每个请求都得到响应
- **P**artition Tolerance（分区容错性）：网络分区时系统仍运行

> 网络分区不可避免，实际上是 **CP vs AP** 的选择。  
> MySQL（主从）→ CP；Cassandra → AP；ZooKeeper → CP。

---

### 2. BASE 理论

对 CAP 中 AP 系统的妥协方案：
- **B**asically Available（基本可用）：允许部分不可用
- **S**oft State（软状态）：数据可以有中间状态
- **E**ventual Consistency（最终一致性）：数据最终一致

---

### 3. 消息队列（MQ）的作用

- **解耦：** 生产者/消费者独立
- **削峰：** 流量高峰时缓冲请求
- **异步：** 非核心链路异步处理

**常见 MQ 对比：**

| | Kafka | RabbitMQ | RocketMQ |
|---|---|---|---|
| 吞吐量 | 极高（百万级/s）| 中（万级）| 高（十万级）|
| 延迟 | 较高（ms 级）| 低（μs 级）| 低 |
| 消费模型 | Pull（消费者拉）| Push | Pull/Push |
| 场景 | 日志、流处理 | 业务消息 | 交易、订单 |

---

### 4. 幂等性设计

相同请求执行多次，结果与执行一次相同。

**实现方案：**
- **唯一请求 ID：** 客户端生成，服务端去重（Redis SET NX）
- **数据库唯一约束：** 防重复插入
- **状态机：** 只允许特定状态转换
- **乐观锁：** 版本号控制

---

### 5. 分布式事务

| 方案 | 说明 | 缺点 |
|---|---|---|
| 2PC（两阶段提交）| 协调者广播准备→提交 | 同步阻塞，单点故障 |
| TCC | Try-Confirm-Cancel，业务层补偿 | 代码侵入性大 |
| Saga | 长事务拆分，失败逐步补偿 | 复杂度高 |
| 本地消息表 | 事务内写消息表，异步消费 | 实现简单，最终一致 |

---

### 6. 限流算法

| 算法 | 说明 | 特点 |
|---|---|---|
| 固定窗口 | 每个时间窗口计数，超过限制 | 临界突刺问题 |
| 滑动窗口 | 精确记录每个请求时间 | 较精确，内存大 |
| 漏桶 | 恒定速率处理，多余丢弃 | 流量平滑，不允许突发 |
| 令牌桶 | 恒定速率生成令牌，有桶容量 | 允许一定突发（常用）|

---

### 7. 接口幂等 & 防重复提交

```
前端：提交后禁用按钮
后端：
  1. 客户端生成 token，提交前获取
  2. 服务端 Redis SET NX token 处理中
  3. 处理完成删除 token
  → 重复请求直接返回"处理中"或"已处理"
```

---

### 8. 秒杀系统设计

```
1. 前端限流：按钮置灰、随机放行
2. 网关限流：令牌桶/漏桶
3. Redis 预扣库存：DECR + 判断 >= 0（原子操作）
4. 异步下单：发 MQ，订单服务消费
5. 数据库：数据库扣减 + 乐观锁（version 字段）
6. 超卖保护：库存不足时快速失败
```

---

---

## 七、Python Web 框架

### 1. Django vs Flask vs FastAPI

| | Django | Flask | FastAPI |
|---|---|---|---|
| 类型 | 全栈重量级 | 微框架 | 现代异步框架 |
| ORM | 内置 Django ORM | 无（用 SQLAlchemy）| 无（用 SQLAlchemy/Tortoise）|
| 异步 | 部分支持 | 有限支持 | ✅ 原生 async/await |
| 性能 | 一般 | 中等 | 极高（接近 Go）|
| 文档 | 手动 | 手动 | ✅ 自动（OpenAPI）|
| 适用 | 快速搭建完整应用 | 灵活小项目 | 高性能 API、微服务 |

---

### 2. ORM 核心概念

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

### 3. RESTful API 设计规范

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

### 4. JWT 认证

```
Header.Payload.Signature

Header: {"alg": "HS256", "typ": "JWT"}
Payload: {"sub": "user_id", "exp": 时间戳}
Signature: HMACSHA256(base64(header)+"."+base64(payload), secret)
```

- 无状态，服务端不存 session
- 缺点：无法主动失效（解决：Redis 黑名单 / refresh token 机制）

---

*文档共计 70+ 题，持续更新。*
