
---

##  基本数据类型

| 类型 | Go | Python | JavaScript |
|------|-----|--------|-----------|
| 整数 | `int` `int8` `int64` `uint` … | `int`（无限精度） | `Number`（IEEE 754）|
| 浮点 | `float32` `float64` | `float` | `Number` |
| 布尔 | `bool` (`true`/`false`) | `bool` (`True`/`False`) | `boolean` (`true`/`false`) |
| 字符串 | `string`（不可变，UTF-8） | `str`（不可变，Unicode） | `string`（不可变，UTF-16） |
| 字节/字符 | `byte`（`uint8`）/ `rune`（`int32`） | — | — |
| 空值 | `nil` | `None` | `null` / `undefined` |
| 复数 | `complex64/128` | `complex` | — |
| 类型查询 | `reflect.TypeOf(x)` | `type(x)` | `typeof x` |

---

##  字符串

| 操作 | Go | Python | JavaScript |
|------|-----|--------|-----------|
| 拼接 | `s1 + s2` | `s1 + s2` | `s1 + s2` |
| 格式化 | `fmt.Sprintf("%d", n)` | `f"{n}"` | `` `${n}` `` |
| 多行字符串 | `` `反引号` `` | `'''三引号'''` | `` `反引号` `` |
| 长度（字节） | `len(s)` | `len(s)` | `s.length` |
| 字符数（Unicode）| `utf8.RuneCountInString(s)` | `len(s)` | `[...s].length` |
| 切片 | `s[1:4]` | `s[1:4]` | `s.slice(1, 4)` |
| 包含 | `strings.Contains(s, sub)` | `sub in s` | `s.includes(sub)` |
| 替换 | `strings.ReplaceAll(s, o, n)` | `s.replace(o, n)` | `s.replaceAll(o, n)` |
| 分割 | `strings.Split(s, ",")` | `s.split(",")` | `s.split(",")` |
| 去空格 | `strings.TrimSpace(s)` | `s.strip()` | `s.trim()` |
| 大小写 | `strings.ToUpper/Lower(s)` | `s.upper() / s.lower()` | `s.toUpperCase() / s.toLowerCase()` |
| 反转 | 手动循环 | `s[::-1]` | `[...s].reverse().join("")` |

---

##  运算符

| 运算符 | Go | Python | JavaScript |
|--------|-----|--------|-----------|
| 整除 | `a / b`（整数自动整除） | `a // b` | `Math.floor(a / b)` |
| 幂运算 | `math.Pow(2, 10)` | `2 ** 10` | `2 ** 10` |
| 逻辑与/或 | `&&` / `\|\|` | `and` / `or` | `&&` / `\|\|` |
| 逻辑非 | `!` | `not` | `!` |
| 空值回退 | — | `x or "default"` | `x ?? "default"` |
| 链式比较 | 不支持 | `1 < x < 10` | 不支持 |
| 位运算 | `&` `\|` `^` `<<` `>>` | `&` `\|` `^` `~` `<<` `>>` | `&` `\|` `^` `~` `<<` `>>` |
| 自增 | `i++` / `i--`（语句，非表达式）| 无 | `i++` / `i--` |
| 严格相等 | `==`（强类型，自动匹配）| `==` / `is` | `===`（推荐）/ `==` |


---

##  数组 / 列表

| 操作 | Go（切片） | Python（list） | JavaScript（Array） |
|------|-----------|---------------|-------------------|
| 创建 | `[]int{1,2,3}` | `[1, 2, 3]` | `[1, 2, 3]` |
| 追加 | `append(sl, 4)` | `lst.append(4)` | `arr.push(4)` |
| 头部插入 | `append([]T{x}, sl...)` | `lst.insert(0, x)` | `arr.unshift(x)` |
| 删除末尾 | `sl[:len(sl)-1]` | `lst.pop()` | `arr.pop()` |
| 删除头部 | `sl[1:]` | `lst.pop(0)` | `arr.shift()` |
| 删除任意索引 i | `append(sl[:i], sl[i+1:]...)` | `del lst[i]` | `arr.splice(i, 1)` |
| 切片 | `sl[1:4]` | `lst[1:4]` | `arr.slice(1, 4)` |
| 长度 | `len(sl)` | `len(lst)` | `arr.length` |
| 排序 | `sort.Ints(sl)` | `lst.sort()` / `sorted(lst)` | `arr.sort((a,b)=>a-b)` |
| 遍历 | `for i, v := range sl` | `for i, v in enumerate(lst)` | `for (const [i, v] of arr.entries())` |
| map | 手动循环 | `list(map(f, lst))` / 推导式 | `arr.map(f)` |
| filter | 手动循环 | `list(filter(f, lst))` / 推导式 | `arr.filter(f)` |
| reduce | 手动循环 | `functools.reduce(f, lst)` | `arr.reduce(f, init)` |
| 拷贝 | `copy(dst, src)` | `lst[:]` / `lst.copy()` | `[...arr]` |
| 合并 | `append(a, b...)` | `a + b` | `[...a, ...b]` |
| 包含 | 手动循环 | `x in lst` | `arr.includes(x)` |
| 查找索引 | 手动循环 | `lst.index(x)` | `arr.indexOf(x)` |

---

##  字典 / Map / Object

| 操作 | Go（map） | Python（dict） | JavaScript（Object / Map） |
|------|----------|--------------|--------------------------|
| 创建 | `map[string]int{"a": 1}` | `{"a": 1}` | `{a: 1}` / `new Map()` |
| 读取 | `m["key"]` | `d["key"]` / `d.get("key")` | `obj.key` / `map.get("key")` |
| 安全读取 | `v, ok := m["key"]` | `d.get("key", default)` | `obj?.key ?? default` |
| 写入 | `m["key"] = v` | `d["key"] = v` | `obj.key = v` / `map.set("k", v)` |
| 删除 | `delete(m, "key")` | `del d["key"]` / `d.pop("key")` | `delete obj.key` / `map.delete("k")` |
| 包含 | `_, ok := m["key"]` | `"key" in d` | `"key" in obj` / `map.has("k")` |
| 遍历 | `for k, v := range m` | `for k, v in d.items()` | `Object.entries(obj)` / `for..of map` |
| 所有键 | 手动 range | `d.keys()` | `Object.keys(obj)` / `[...map.keys()]` |
| 合并 | 手动 range | `d1 \| d2`（3.9+） | `{...obj1, ...obj2}` |
| 长度 | `len(m)` | `len(d)` | `Object.keys(obj).length` / `map.size` |

---

##  集合 Set

| 操作 | Go（用 map 模拟） | Python（set） | JavaScript（Set） |
|------|----------------|-------------|-----------------|
| 创建 | `map[T]struct{}{}` | `{1, 2, 3}` / `set([...])` | `new Set([1,2,3])` |
| 添加 | `s[v] = struct{}{}` | `s.add(v)` | `s.add(v)` |
| 删除 | `delete(s, v)` | `s.discard(v)` | `s.delete(v)` |
| 包含 | `_, ok := s[v]` | `v in s` | `s.has(v)` |
| 交集 | 手动 | `a & b` | `new Set([...a].filter(x=>b.has(x)))` |
| 并集 | 手动 | `a \| b` | `new Set([...a, ...b])` |
| 差集 | 手动 | `a - b` | `new Set([...a].filter(x=>!b.has(x)))` |
| 长度 | `len(s)` | `len(s)` | `s.size` |

---

##  控制流

### switch / match

| | Go | Python | JavaScript |
|-|-----|--------|-----------|
| 语法 | `switch x { case 1: ... }` | `match x:` (3.10+) | `switch(x) { case 1: ...; break; }` |
| 无需 break | ✅ 默认不穿透 | ✅ | ❌ 需要显式 break |
| 表达式 switch | ✅ `switch { case x>0: }` | ✅ 条件 `case` | ❌ |
| 多值匹配 | `case 1, 2:` | `case 1 \| 2:` | `case 1: case 2:` |

### for 循环

| 形式 | Go | Python | JavaScript |
|------|-----|--------|-----------|
| 计数 | `for i := 0; i < n; i++` | `for i in range(n)` | `for (let i = 0; i < n; i++)` |
| while 等价 | `for k < 50` | `while k < 50` | `while (k < 50)` |
| 遍历集合 | `for i, v := range sl` | `for i, v in enumerate(lst)` | `for (const [i, v] of arr.entries())` |
| 遍历 map | `for k, v := range m` | `for k, v in d.items()` | `for (const [k, v] of Object.entries(obj))` |
| 并行遍历 | 手动 index | `zip(a, b)` | `a.forEach((v, i) => ...)` |
| break / continue | `break` / `continue` | `break` / `continue` | `break` / `continue` |
| 带标签跳转 | `break outer` | — | `break outer` |
| for…else | ❌ | ✅ | ❌ |

---

## 9. 函数

| 特性 | Go | Python | JavaScript |
|------|-----|--------|-----------|
| 定义 | `func add(a, b int) int` | `def add(a, b):` | `function add(a, b) {` |
| 多返回值 | ✅ `return a, b` | ✅ `return a, b`（元组） | ❌（需返回对象/数组） |
| 命名返回值 | ✅ `func f() (res int)` | ❌ | ❌ |
| 默认参数 | ❌（用变参模拟） | ✅ `def f(x=0)` | ✅ `function f(x=0)` |
| 关键字参数 | ❌ | ✅ `f(key=val)` | ✅（解构：`f({key})`) |
| 可变参数 | `nums ...int` | `*args` | `...nums` |
| 函数类型 | `func(int) int` | `Callable[[int], int]` | `(x: number) => number` |

---

## 10. 闭包

```go
// Go
makeCounter := func() func() int {
    count := 0
    return func() int { count++; return count }
}
counter := makeCounter()
fmt.Println(counter(), counter(), counter())  // 1 2 3
```

```python
# Python
def make_counter():
    count = 0
    def counter():
        nonlocal count  # 修改外层变量需声明
        count += 1
        return count
    return counter

counter = make_counter()
print(counter(), counter(), counter())  # 1 2 3

# 装饰器本质是闭包
import functools
def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import time; start = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__} 耗时 {time.perf_counter()-start:.4f}s")
        return result
    return wrapper
```

```js
// JavaScript
const makeCounter = () => {
  let count = 0;
  return () => ++count;
};
const counter = makeCounter();
console.log(counter(), counter(), counter());  // 1 2 3
```

---

## 11. 面向对象

### 基本结构

| 概念 | Go | Python | JavaScript |
|------|-----|--------|-----------|
| 类/结构体 | `struct` | `class` | `class` |
| 构造器 | 普通函数 `NewXxx()` | `__init__` | `constructor` |
| 实例方法 | `func (r Receiver) Method()` | `def method(self)` | `method() {}` |
| 静态方法 | 包级函数 | `@staticmethod` | `static method() {}` |
| 类方法 | — | `@classmethod` | `static` |
| 继承 | 嵌入（组合）| `class Dog(Animal)` | `class Dog extends Animal` |
| 调用父类 | `animal.Method()` | `super().__init__()` | `super()` |
| 私有成员 | 小写字母开头 | 约定 `_` 前缀 | `#privateField`（ES2022） |
| 接口 | `interface` | `Protocol` / ABC | — |
| 多态 | 接口实现 | 继承 + 方法重写 | 继承 + 方法重写 |

```go
// Go
type Animal struct { Name string }
func (a Animal) Speak() string { return "..." }

type Dog struct {
    Animal         // 嵌入（组合替代继承）
    Breed string
}
func (d Dog) Speak() string { return d.Name + " 汪汪!" }

d := Dog{Animal: Animal{"旺财"}, Breed: "拉布拉多"}
fmt.Println(d.Speak())  // 旺财 汪汪!
fmt.Println(d.Name)     // 直接访问嵌入字段
```

```python
# Python
class Animal:
    kingdom = "Animalia"  # 类变量

    def __init__(self, name, sound):
        self.name = name
        self._sound = sound  # 约定保护

    def speak(self):
        return f"{self.name} says {self._sound}"

    @classmethod
    def from_dict(cls, data):
        return cls(data["name"], data["sound"])

    @staticmethod
    def is_valid_name(name):
        return bool(name and name.isalpha())

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name, "Woof")
        self.breed = breed

    def speak(self):
        return super().speak() + f" (breed: {self.breed})"
```
---

## 13. 错误处理

| 特性 | Go | Python | JavaScript |
|------|-----|--------|-----------|
| 错误传递 | 返回值 `(T, error)` | 异常 `raise` | 异常 `throw` |
| 捕获 | `if err != nil` | `try / except` | `try / catch` |
| 清理 | `defer` | `finally` / `with` | `finally` |
| 自定义错误 | 实现 `Error() string` | 继承 `Exception` | 继承 `Error` |
| 错误包装 | `fmt.Errorf("...: %w", err)` | `raise X from Y` | `new Error(msg, {cause: err})` |
| 错误检查 | `errors.Is` / `errors.As` | `isinstance(e, Type)` | `e instanceof Type` |
| 恐慌/崩溃 | `panic` / `recover` | `raise` / `except` | `throw` / `catch` |

```go
// Go：错误是返回值，必须显式处理
func divide(a, b float64) (float64, error) {
    if b == 0 { return 0, errors.New("除数不能为 0") }
    return a / b, nil
}
res, err := divide(10, 0)
if err != nil { fmt.Println("错误:", err) }

// 错误包装与检查
var ErrNotFound = errors.New("not found")
err := fmt.Errorf("查询失败: %w", ErrNotFound)
if errors.Is(err, ErrNotFound) { fmt.Println("未找到") }
```

```python
# Python：异常机制
def divide(a, b):
    if b == 0:
        raise ZeroDivisionError("除数不能为 0")
    return a / b

try:
    result = divide(10, 0)
except ZeroDivisionError as e:
    print("错误:", e)
except TypeError as e:
    print("类型错误:", e)
else:
    print("成功:", result)  # 无异常时执行
finally:
    print("always")         # 总是执行

# 自定义异常
class ValidationError(Exception):
    def __init__(self, field, message):
        self.field = field
        super().__init__(f"[{field}] {message}")
```

---

## 14. 并发

| 特性 | Go | Python | JavaScript |
|------|-----|--------|-----------|
| 并发模型 | Goroutine + Channel（CSP） | threading / asyncio | 事件循环 + Promise / async-await |
| 启动并发 | `go func()` | `threading.Thread` / `asyncio.create_task` | `Promise` / `async` |
| 等待完成 | `sync.WaitGroup` | `t.join()` / `await gather()` | `Promise.all()` / `await` |
| 通信 | `chan`（Channel） | `queue.Queue` / `asyncio.Queue` | — |
| 锁 | `sync.Mutex` | `threading.Lock` | — |
| 原子操作 | `sync/atomic` | — | — |
| GIL 限制 | ❌（真正并行） | ✅（CPU 密集受限）| ❌（单线程事件循环） |
| 适合场景 | CPU 密集 + IO 密集 | IO 密集（asyncio） | IO 密集（异步） |

```go
// Go：Goroutine + WaitGroup + Channel
var wg sync.WaitGroup
for i := 0; i < 3; i++ {
    wg.Add(1)
    go func(id int) {
        defer wg.Done()
        fmt.Printf("goroutine %d\n", id)
    }(i)
}
wg.Wait()

// Channel 通信
ch := make(chan int, 1)
go func() { ch <- 42 }()
fmt.Println(<-ch)

// select 多路复用
select {
case msg := <-ch1: fmt.Println("ch1:", msg)
case msg := <-ch2: fmt.Println("ch2:", msg)
case <-time.After(1 * time.Second): fmt.Println("超时")
}
```

```python
# Python：asyncio（IO 密集首选）
import asyncio

async def fetch(id, delay):
    await asyncio.sleep(delay)
    return f"task {id} done"

async def main():
    results = await asyncio.gather(
        fetch(1, 0.3),
        fetch(2, 0.1),
        fetch(3, 0.2),
    )
    for r in results: print(r)

# threading（受 GIL 限制，CPU 密集用 ProcessPoolExecutor）
import threading
lock = threading.Lock()
counter = 0
def increment():
    global counter
    with lock: counter += 1
```