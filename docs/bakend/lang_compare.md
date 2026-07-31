# Go / Python / JavaScript 语法对比

> 覆盖变量、数据结构、控制流、函数、面向对象、错误处理、并发等核心主题。

---

## 目录

1. [变量声明](#1-变量声明)
2. [基本数据类型](#2-基本数据类型)
3. [字符串](#3-字符串)
4. [运算符](#4-运算符)
5. [数组 / 列表](#5-数组--列表)
6. [字典 / Map / Object](#6-字典--map--object)
7. [集合 Set](#7-集合-set)
8. [控制流](#8-控制流)
9. [函数](#9-函数)
10. [闭包](#10-闭包)
11. [面向对象](#11-面向对象)
12. [接口 / 协议 / 鸭子类型](#12-接口--协议--鸭子类型)
13. [错误处理](#13-错误处理)
14. [并发](#14-并发)
15. [类型系统](#15-类型系统)
16. [常用标准库速查](#16-常用标准库速查)

---

## 1. 变量声明

| 特性 | Go | Python | JavaScript |
|------|-----|--------|-----------|
| 显式声明 | `var x int = 42` | —（无需声明类型） | `let x = 42` |
| 简短声明 | `x := 42` | `x = 42` | `const x = 42` |
| 常量 | `const Pi = 3.14` | `PI = 3.14`（约定大写） | `const PI = 3.14` |
| 多重赋值 | `a, b := 1, 2` | `a, b = 1, 2` | `let [a, b] = [1, 2]` |
| 变量交换 | `a, b = b, a` | `a, b = b, a` | `[a, b] = [b, a]` |
| 零值 / 默认值 | 有（`0` `""` `false` `nil`） | 无（未赋值会报错） | `undefined` |
| 作用域关键字 | — | `global` / `nonlocal` | `var` / `let` / `const` |

```go
// Go
var count int = 42
score := 88
const Tau = 6.28
a, b := 1, 2
a, b = b, a
```

```python
# Python
count = 42
score = 88
TAU = 6.28
a, b = 1, 2
a, b = b, a
first, *rest = [1, 2, 3, 4]  # 解包
```

```js
// JavaScript
let count = 42;
const TAU = 6.28;
let [first, ...rest] = [1, 2, 3, 4];  // 解构
let a = 1, b = 2;
[a, b] = [b, a];
```

---

## 2. 基本数据类型

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

```go
// Go
var i8 int8 = 127
var b byte = 'A'
var r rune = '文'
var z complex128 = complex(1, 2)
```

```python
# Python
i = 42          # int，无上限
f = 3.14
c = 2 + 3j      # 复数
print(type(i))  # <class 'int'>
```

```js
// JavaScript
console.log(typeof 42);          // "number"
console.log(typeof "hi");        // "string"
console.log(typeof undefined);   // "undefined"
console.log(typeof null);        // "object"（历史遗留 bug）
console.log(Number.MAX_SAFE_INTEGER);
```

---

## 3. 字符串

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

```go
// Go
import (
    "fmt"
    "strings"
    "unicode/utf8"
)
msg := "Hello, 世界"
fmt.Println(len(msg))                          // 字节数 13
fmt.Println(utf8.RuneCountInString(msg))       // 字符数 9
fmt.Println(strings.Contains(msg, "世界"))     // true
fmt.Sprintf("score=%d, ratio=%.2f", 88, 3.14) // 格式化
```

```python
# Python
msg = "Hello, 世界"
print(len(msg))              # 9（字符数）
print("世界" in msg)         # True
print(f"score={88}, ratio={3.14:.2f}")
print(msg[::-1])             # 反转
```

```js
// JavaScript
const msg = "Hello, 世界";
console.log(msg.length);              // 9
console.log(msg.includes("世界"));   // true
console.log(`score=${88}`);          // 模板字符串
console.log([...msg].reverse().join(""));  // 反转
```

---

## 4. 运算符

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

```go
// Go
fmt.Println(7 / 2)      // 3（整数整除）
fmt.Println(7.0 / 2)    // 3.5
i := 5; i++; fmt.Println(i)  // 6
```

```python
# Python
print(7 / 2)    # 3.5（真除法）
print(7 // 2)   # 3（整除）
print(2 ** 10)  # 1024
print(1 < 2 < 3)  # True（链式比较）
x = None
print(x or "default")   # default
```

```js
// JavaScript
console.log(7 / 2);          // 3.5
console.log(Math.floor(7/2)); // 3
console.log(2 ** 10);         // 1024
console.log(null ?? "default");  // "default"
console.log(0 ?? "default");     // 0（仅 null/undefined 触发）
console.log(0 || "default");     // "default"（假值都触发）
```

---

## 5. 数组 / 列表

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

```go
// Go
sl := []int{3, 1, 4, 1, 5}
sl = append(sl, 9)
part := sl[1:3]
sl = append(sl[:1], sl[2:]...)  // 删除索引 1

// 推导式等价：手动循环
squares := make([]int, len(sl))
for i, v := range sl { squares[i] = v * v }
```

```python
# Python
lst = [3, 1, 4, 1, 5]
lst.append(9)
part = lst[1:3]
del lst[1]

# 推导式（最 Pythonic）
squares = [x**2 for x in lst]
evens   = [x for x in lst if x % 2 == 0]
```

```js
// JavaScript
const arr = [3, 1, 4, 1, 5];
arr.push(9);
const part = arr.slice(1, 3);
arr.splice(1, 1);  // 删除索引 1

const squares = arr.map(x => x ** 2);
const evens   = arr.filter(x => x % 2 === 0);
const total   = arr.reduce((acc, x) => acc + x, 0);
```

---

## 6. 字典 / Map / Object

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

```go
// Go
store := map[string]int{"a": 1, "b": 2}
store["c"] = 3
delete(store, "b")
if v, ok := store["a"]; ok {
    fmt.Println(v)
}
for k, v := range store { fmt.Println(k, v) }
```

```python
# Python
d = {"a": 1, "b": 2}
d["c"] = 3
del d["b"]
print(d.get("x", 0))  # 安全读取，默认 0
d.update({"d": 4, "e": 5})
merged = d | {"f": 6}  # 3.9+

for k, v in d.items():
    print(k, v)
```

```js
// JavaScript
// Object（键只能是字符串/Symbol）
const obj = {a: 1, b: 2};
obj.c = 3;
delete obj.b;
console.log(Object.keys(obj), Object.values(obj));
const merged = {...obj, d: 4};

// Map（键可以是任意类型）
const map = new Map([["a", 1], ["b", 2]]);
map.set("c", 3);
map.delete("b");
for (const [k, v] of map) console.log(k, v);
```

---

## 7. 集合 Set

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

```python
# Python
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}
print(a & b)   # {3, 4}  交集
print(a | b)   # {1,2,3,4,5,6} 并集
print(a - b)   # {1, 2}  差集
print(a ^ b)   # {1,2,5,6} 对称差
```

```js
// JavaScript
const a = new Set([1, 2, 3, 4]);
const b = new Set([3, 4, 5, 6]);
const inter = new Set([...a].filter(x => b.has(x)));  // {3, 4}
const union = new Set([...a, ...b]);
```

---

## 8. 控制流

### if / else

```go
// Go（if 可带初始化语句）
if n := 10; n%2 == 0 {
    fmt.Println("偶数")
} else {
    fmt.Println("奇数")
}
```

```python
# Python（三元表达式）
label = "偶数" if n % 2 == 0 else "奇数"
```

```js
// JavaScript（三元 / 可选链）
const label = n % 2 === 0 ? "偶数" : "奇数";
```

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

```go
// Go
func div(a, b int) (int, int) { return a / b, a % b }
func sum(nums ...int) int {
    total := 0
    for _, n := range nums { total += n }
    return total
}
q, r := div(17, 5)
fmt.Println(sum(1, 2, 3, 4, 5))
```

```python
# Python
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

def func(*args, **kwargs):
    print(args, kwargs)

def connect(host, *, port=8080):  # * 后面强制关键字参数
    print(f"{host}:{port}")

a, b = 10, 3  # 多返回值：其实是元组拆包
```

```js
// JavaScript
function greet(name, greeting = "Hello") {
  return `${greeting}, ${name}!`;
}

function sum(...nums) {
  return nums.reduce((acc, x) => acc + x, 0);
}

// 解构参数
function connect({ host, port = 8080 }) {
  console.log(`${host}:${port}`);
}
```

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

```js
// JavaScript
class Animal {
  constructor(name, sound) {
    this.name = name;
    this.sound = sound;
  }
  speak() { return `${this.name} says ${this.sound}`; }
  static create(name, sound) { return new Animal(name, sound); }
}

class Dog extends Animal {
  constructor(name, breed) {
    super(name, "Woof");
    this.breed = breed;
  }
  speak() { return super.speak() + ` (breed: ${this.breed})`; }
}
```

### 魔术方法 / 特殊方法

| 用途 | Go | Python | JavaScript |
|------|-----|--------|-----------|
| 字符串表示 | 实现 `String() string` | `__str__` / `__repr__` | `toString()` |
| 相等比较 | `==`（值语义） | `__eq__` | `[Symbol.toPrimitive]` |
| 运算符重载 | ❌ | `__add__` `__mul__` 等 | ❌ |
| 迭代 | 实现 `range` 函数 | `__iter__` / `__next__` | `[Symbol.iterator]` |
| 长度 | `len()` 函数 | `__len__` | — |
| 调用 | ❌ | `__call__` | — |
| 上下文管理 | `defer` | `__enter__` / `__exit__` | — |

---

## 12. 接口 / 协议 / 鸭子类型

```go
// Go：隐式接口，结构体无需显式声明实现
type Shape interface {
    Area() float64
    Perimeter() float64
}

type Circle struct{ Radius float64 }
func (c Circle) Area() float64      { return math.Pi * c.Radius * c.Radius }
func (c Circle) Perimeter() float64 { return 2 * math.Pi * c.Radius }

// Circle 自动实现 Shape，无需 implements 关键字
var s Shape = Circle{Radius: 5}
```

```python
# Python：Protocol（结构子类型，3.8+）
from typing import Protocol, runtime_checkable

@runtime_checkable
class Drawable(Protocol):
    def draw(self) -> str: ...

class Circle:
    def draw(self) -> str: return "○"

# 无需继承，duck typing
print(isinstance(Circle(), Drawable))  # True

# 也可用抽象基类（强制）
from abc import ABC, abstractmethod
class Shape(ABC):
    @abstractmethod
    def area(self) -> float: ...
```

```js
// JavaScript：没有原生接口，用约定或 TypeScript
// 鸭子类型：只要有相应方法即可
class Circle {
  constructor(r) { this.r = r; }
  area() { return Math.PI * this.r ** 2; }
}
class Square {
  constructor(s) { this.s = s; }
  area() { return this.s ** 2; }
}
// 多态调用
[new Circle(5), new Square(4)].forEach(s => console.log(s.area()));
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

```js
// JavaScript：try/catch
function divide(a, b) {
  if (b === 0) throw new Error("除数不能为 0");
  return a / b;
}

try {
  const result = divide(10, 0);
} catch (e) {
  console.log("错误:", e.message);
} finally {
  console.log("always");
}

// 自定义错误
class ValidationError extends Error {
  constructor(field, message) {
    super(`[${field}] ${message}`);
    this.field = field;
    this.name = "ValidationError";
  }
}
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

```js
// JavaScript：Promise + async/await
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

// 并发执行
const results = await Promise.all([
  delay(300).then(() => "task 1"),
  delay(100).then(() => "task 2"),
  delay(200).then(() => "task 3"),
]);
console.log(results);

// async/await
async function main() {
  const data = await fetch("/api/data").then(r => r.json());
  return data;
}
```

---

## 15. 类型系统

| 特性 | Go | Python | JavaScript |
|------|-----|--------|-----------|
| 类型 | 静态强类型 | 动态强类型 | 动态弱类型 |
| 类型推导 | ✅（`:=`） | ✅（自动） | ✅（自动） |
| 类型注解 | 必须写 | `x: int`（可选，3.5+） | TypeScript 扩展 |
| 泛型 | ✅（1.18+） | ✅（TypeVar） | ✅（TypeScript） |
| 类型断言 | `x.(T)` `x.(type)` | `isinstance(x, T)` | `x instanceof T` |
| 空安全 | `nil` 需手动检查 | `None` 需手动检查 | `?.` 可选链（ES2020） |
| 枚举 | `iota` 常量组 | `Enum` 类 | 约定 `Object.freeze({})` |

```go
// Go：泛型（1.18+）
func First[T any](s []T) T { return s[0] }
func Sum[T ~int | ~float64](s []T) T {
    var total T
    for _, v := range s { total += v }
    return total
}
```

```python
# Python：类型注解 + 泛型
from typing import TypeVar, Optional
T = TypeVar("T")

def first(lst: list[T]) -> Optional[T]:
    return lst[0] if lst else None

def process(data: list[int]) -> dict[str, float]:
    return {"mean": sum(data)/len(data)}
```

```ts
// TypeScript（JavaScript 超集）：静态类型
function first<T>(arr: T[]): T | undefined {
  return arr[0];
}

function sum(nums: number[]): number {
  return nums.reduce((a, b) => a + b, 0);
}
```

---

## 16. 常用标准库速查

### 数学

| 操作 | Go | Python | JavaScript |
|------|-----|--------|-----------|
| 平方根 | `math.Sqrt(x)` | `math.sqrt(x)` | `Math.sqrt(x)` |
| 幂 | `math.Pow(x, y)` | `x ** y` | `x ** y` |
| 绝对值 | `math.Abs(x)` | `abs(x)` | `Math.abs(x)` |
| 最大/最小 | `math.Max/Min(a,b)` | `max(a, b)` | `Math.max(a, b)` |
| 取整 | `math.Floor/Ceil/Round` | `math.floor/ceil/round` | `Math.floor/ceil/round` |
| 随机数 | `math/rand` | `random.random()` | `Math.random()` |
| 圆周率 | `math.Pi` | `math.pi` | `Math.PI` |

### 排序

```go
// Go
sort.Ints([]int{3, 1, 2})
sort.Strings([]string{"b", "a"})
sort.Slice(people, func(i, j int) bool { return people[i].Age < people[j].Age })
```

```python
# Python
sorted([3, 1, 2])                          # 不可变，返回新列表
[3, 1, 2].sort()                           # 原地排序
sorted(people, key=lambda p: p.age)
```

```js
// JavaScript
[3, 1, 2].sort((a, b) => a - b);          // 原地排序
[...arr].sort((a, b) => a - b);           // 拷贝后排序
people.sort((a, b) => a.age - b.age);
```

### 正则表达式

| 操作 | Go | Python | JavaScript |
|------|-----|--------|-----------|
| 编译 | `regexp.MustCompile(pattern)` | `re.compile(pattern)` | `/pattern/flags` |
| 匹配 | `re.MatchString(s)` | `re.match(p, s)` | `re.test(s)` |
| 查找 | `re.FindString(s)` | `re.search(p, s)` | `s.match(re)` |
| 全部 | `re.FindAllString(s, -1)` | `re.findall(p, s)` | `s.match(/re/g)` |
| 替换 | `re.ReplaceAllString(s, repl)` | `re.sub(p, repl, s)` | `s.replace(/re/g, repl)` |

### 时间日期

```go
// Go
now := time.Now()
fmt.Println(now.Format("2006-01-02 15:04:05"))  // 固定参考时间
t, _ := time.Parse("2006-01-02", "2026-03-16")
fmt.Println(now.Add(24 * time.Hour))
```

```python
# Python
from datetime import datetime, timedelta
now = datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))
t = datetime.strptime("2026-03-16", "%Y-%m-%d")
print(now + timedelta(days=30))
```

```js
// JavaScript
const now = new Date();
console.log(now.toISOString());
console.log(now.toLocaleDateString("zh-CN"));
const tomorrow = new Date(now.getTime() + 86400000);
```

### JSON

```go
// Go
import "encoding/json"
data, _ := json.Marshal(obj)
json.Unmarshal(data, &target)
// 结构体标签
type User struct {
    ID   int    `json:"id"`
    Name string `json:"name,omitempty"`
}
```

```python
# Python
import json
s = json.dumps({"name": "Alice"}, ensure_ascii=False, indent=2)
obj = json.loads(s)
```

```js
// JavaScript
const s = JSON.stringify({name: "Alice"}, null, 2);
const obj = JSON.parse(s);
```

### 文件 I/O

```go
// Go
os.WriteFile("file.txt", []byte("hello"), 0644)
data, _ := os.ReadFile("file.txt")
```

```python
# Python
from pathlib import Path
Path("file.txt").write_text("hello", encoding="utf-8")
content = Path("file.txt").read_text()

with open("file.txt") as f:
    for line in f:
        print(line.rstrip())
```

```js
// JavaScript（Node.js）
const fs = require("fs");
fs.writeFileSync("file.txt", "hello");
const content = fs.readFileSync("file.txt", "utf8");
// 或 async
await fs.promises.writeFile("file.txt", "hello");
```

---

## 附：关键差异一览

| 维度 | Go | Python | JavaScript |
|------|-----|--------|-----------|
| **范式** | 命令式、并发、少OOP | 多范式（OOP+函数式+命令式）| 多范式（OOP+函数式+事件驱动）|
| **编译** | 静态编译 | 解释执行 | JIT 编译（V8 等）|
| **内存管理** | GC | GC（引用计数+循环检测）| GC |
| **包管理** | Go Modules | pip / conda | npm / yarn / pnpm |
| **格式化** | `gofmt`（官方强制）| `black` / `ruff` | `prettier` |
| **错误哲学** | 错误是值，显式处理 | 异常驱动 | 异常驱动 |
| **并发哲学** | CSP（通过通信共享内存）| GIL（asyncio 绕开）| 单线程事件循环 |
| **零值哲学** | 类型有默认零值 | 未赋值即报错 | `undefined` |
| **接口哲学** | 隐式满足（duck typing）| 显式继承或 Protocol | 约定（TypeScript 补充）|

---

## 附：综合实战函数 `analyzeText`

> 设计一个**文本词频分析**函数，在单个函数实现中尽量覆盖三语言的核心语法差异。

### 函数设计

**输入**：一段文本 + 可选配置（最短词长、topN、排除词表）  
**输出**：`{ total, unique, topWords, avgLen, longestWord }` 结构体 / dataclass / 对象  
**边界**：空文本 / 过滤后无词 → 错误处理

### 语法覆盖矩阵

| 知识点 | 代码位置 | Go | Python | JavaScript |
|--------|----------|----|--------|-----------|
| 自定义数据结构 | 返回结果 | `struct` | `@dataclass` | 普通对象字面量 |
| 可选/默认参数 | 函数签名 | Functional Options | `*` 强制关键字参数 | 解构默认值 |
| 错误处理 | 空输入检查 | `(T, error)` 返回值 | `raise` 自定义异常 | `throw` 自定义 Error |
| 闭包 | `isValid` 过滤器 | 匿名函数捕获 `cfg` | `lambda` 捕获外层变量 | 箭头函数捕获 `excludeSet` |
| 字符串操作 | 分词规范化 | `strings` 包 | `re.sub` + `str.split` | `replace` + `split` |
| 列表推导 / 过滤 | 提取有效词 | `for range` + `append` | 列表推导式 `[x for x if]` | `.filter()` 链式调用 |
| Map / 字典 / 频率统计 | 词频统计 | `map[string]int` | `collections.Counter` | `Map` + `reduce` |
| 自定义排序 | topN 排序 | `sort.Slice` 多键 | `sorted(key=lambda)` | `.sort([wa,ca],[wb,cb])` |
| 聚合计算 | 平均/最长 | `for range` 累加 | `sum()` / `max(key=)` | `.reduce()` |
| 类型注解 | 函数签名 | 静态类型（必须） | `hint: type`（可选）| JSDoc 注释 |
| 接口 / 协议 | `Stringer` | 实现 `String()` | `__str__` | `toString()` |

---

### Go 实现

```go
package main

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
)

// ── 数据结构 ──────────────────────────────────────────────

type WordStat struct {
	Word  string
	Count int
}

func (w WordStat) String() string { return fmt.Sprintf("%s(%d)", w.Word, w.Count) }

type AnalysisResult struct {
	Total       int
	Unique      int
	TopWords    []WordStat
	AvgLen      float64
	LongestWord string
}

func (r AnalysisResult) String() string {
	return fmt.Sprintf("total=%d unique=%d avgLen=%.2f longest=%q top=%v",
		r.Total, r.Unique, r.AvgLen, r.LongestWord, r.TopWords)
}

// ── Functional Options（可选参数模式） ───────────────────────

type config struct {
	minLen  int
	topN    int
	exclude map[string]struct{}
}

type Option func(*config)

func WithMinLen(n int) Option  { return func(c *config) { c.minLen = n } }
func WithTopN(n int) Option    { return func(c *config) { c.topN = n } }
func WithExclude(words ...string) Option {
	return func(c *config) {
		for _, w := range words {
			c.exclude[w] = struct{}{}
		}
	}
}

// ── 核心函数 ──────────────────────────────────────────────

func AnalyzeText(text string, opts ...Option) (AnalysisResult, error) {
	// 默认配置
	cfg := &config{minLen: 3, topN: 5, exclude: make(map[string]struct{})}
	for _, opt := range opts {
		opt(cfg)
	}

	// 1. 错误处理：空输入
	if strings.TrimSpace(text) == "" {
		return AnalysisResult{}, errors.New("text cannot be empty")
	}

	// 2. 分词 + 规范化
	//    闭包：isValid 捕获 cfg，复用过滤逻辑
	isValid := func(w string) bool {
		_, excluded := cfg.exclude[w]
		return len([]rune(w)) >= cfg.minLen && !excluded
	}

	raw := strings.Fields(strings.ToLower(text))
	var filtered []string
	for _, w := range raw {
		w = strings.Trim(w, `.,!?;:"'`)
		if isValid(w) {
			filtered = append(filtered, w)
		}
	}

	if len(filtered) == 0 {
		return AnalysisResult{}, errors.New("no valid words after filtering")
	}

	// 3. 词频统计（map）
	freq := make(map[string]int)
	for _, w := range filtered {
		freq[w]++
	}

	// 4. 自定义多键排序：频率降序，同频则字母升序
	pairs := make([]WordStat, 0, len(freq))
	for w, c := range freq {
		pairs = append(pairs, WordStat{w, c})
	}
	sort.Slice(pairs, func(i, j int) bool {
		if pairs[i].Count != pairs[j].Count {
			return pairs[i].Count > pairs[j].Count
		}
		return pairs[i].Word < pairs[j].Word
	})

	topN := int(math.Min(float64(cfg.topN), float64(len(pairs))))

	// 5. 聚合：平均词长 + 最长词（for range 累加）
	totalLen := 0
	longest := ""
	for _, w := range filtered {
		totalLen += len([]rune(w))
		if len([]rune(w)) > len([]rune(longest)) {
			longest = w
		}
	}

	return AnalysisResult{
		Total:       len(filtered),
		Unique:      len(freq),
		TopWords:    pairs[:topN],
		AvgLen:      float64(totalLen) / float64(len(filtered)),
		LongestWord: longest,
	}, nil
}

// ── 主程序 ────────────────────────────────────────────────

func main() {
	text := `Go is an open source programming language that makes it easy
             to build simple reliable and efficient software go go go`

	result, err := AnalyzeText(text,
		WithMinLen(3),
		WithTopN(3),
		WithExclude("and", "that"),
	)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	fmt.Println(result)
	// total=13 unique=11 avgLen=5.38 longest="programming" top=[go(4) simple(1) source(1)]
}
```

---

### Python 实现

```python
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional


# ── 数据结构（dataclass 自动生成 __init__ / __repr__ / __eq__） ─

@dataclass
class WordStat:
    word: str
    count: int

    def __repr__(self) -> str:
        return f"{self.word}({self.count})"


@dataclass
class AnalysisResult:
    total:        int
    unique:       int
    top_words:    list[WordStat]
    avg_len:      float
    longest_word: str

    def __str__(self) -> str:
        return (
            f"total={self.total} unique={self.unique} "
            f"avg_len={self.avg_len:.2f} longest={self.longest_word!r} "
            f"top={self.top_words}"
        )


# ── 自定义异常（继承链） ─────────────────────────────────────

class AnalysisError(Exception):
    pass


# ── 核心函数 ──────────────────────────────────────────────

def analyze_text(
    text: str,
    *,                                     # * 之后强制使用关键字参数
    min_len: int = 3,
    top_n:   int = 5,
    exclude: Optional[set[str]] = None,
) -> AnalysisResult:
    """分析文本，返回词频统计结果。"""
    exclude = exclude or set()

    # 1. 错误处理：空输入
    if not text or not text.strip():
        raise AnalysisError("text cannot be empty")

    # 2. 分词 + 规范化
    #    lambda 闭包：捕获 min_len / exclude，表达单行过滤条件
    is_valid = lambda w: len(w) >= min_len and w not in exclude

    cleaned = re.sub(r"[.,!?;:'\"]+", "", text.lower())
    filtered = [w for w in cleaned.split() if is_valid(w)]  # 列表推导

    if not filtered:
        raise AnalysisError("no valid words after filtering")

    # 3. 词频统计（Counter 是 dict 子类）
    freq = Counter(filtered)

    # 4. 自定义多键排序：频率降序，同频字母升序
    top_words = [
        WordStat(w, c)
        for w, c in sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:top_n]
    ]

    # 5. 聚合：内置函数 sum / max + key 参数
    avg_len = sum(len(w) for w in filtered) / len(filtered)  # 生成器表达式
    longest = max(filtered, key=len)

    return AnalysisResult(
        total=len(filtered),
        unique=len(freq),
        top_words=top_words,
        avg_len=round(avg_len, 2),
        longest_word=longest,
    )


# ── 主程序 ────────────────────────────────────────────────

if __name__ == "__main__":
    text = """Go is an open source programming language that makes it easy
              to build simple reliable and efficient software go go go"""

    try:
        result = analyze_text(
            text,
            min_len=3,
            top_n=3,
            exclude={"and", "that"},
        )
        print(result)
        # total=13 unique=11 avg_len=5.38 longest='programming' top=[go(4), simple(1), source(1)]
    except AnalysisError as e:
        print(f"Error: {e}")
```

---

### JavaScript 实现

```js
// ── 自定义错误 ────────────────────────────────────────────

class AnalysisError extends Error {
  constructor(message) {
    super(message);
    this.name = "AnalysisError";
  }
}

// ── 核心函数 ──────────────────────────────────────────────

/**
 * 分析文本，返回词频统计结果。
 * @param {string} text
 * @param {{ minLen?: number, topN?: number, exclude?: string[] }} [options]
 * @returns {{ total: number, unique: number, topWords: {word,count}[], avgLen: number, longestWord: string }}
 */
function analyzeText(text, { minLen = 3, topN = 5, exclude = [] } = {}) {
  // 解构赋值 + 默认值覆盖可选配置

  const excludeSet = new Set(exclude);   // Set 查找 O(1)

  // 1. 错误处理：空输入（可选链 ?. 防止 null）
  if (!text?.trim()) throw new AnalysisError("text cannot be empty");

  // 2. 分词 + 规范化
  //    箭头函数闭包：isValid 捕获 minLen / excludeSet
  const isValid = (w) => w.length >= minLen && !excludeSet.has(w);

  const filtered = text
    .toLowerCase()
    .replace(/[.,!?;:'"]+/g, "")         // 正则去标点
    .split(/\s+/)
    .filter(Boolean)                      // 去空串
    .filter(isValid);                     // 链式 filter

  if (!filtered.length) throw new AnalysisError("no valid words after filtering");

  // 3. 词频统计（Map + reduce）
  const freq = filtered.reduce((map, w) => {
    map.set(w, (map.get(w) ?? 0) + 1);   // ?? 空值合并
    return map;
  }, new Map());

  // 4. 自定义多键排序：频率降序，同频字母升序
  const topWords = [...freq.entries()]
    .sort(([wa, ca], [wb, cb]) => cb - ca || wa.localeCompare(wb))
    .slice(0, topN)
    .map(([word, count]) => ({ word, count }));  // 解构 + 对象简写

  // 5. 聚合：reduce 求平均，reduce 求最长
  const avgLen = +(
    filtered.reduce((sum, w) => sum + w.length, 0) / filtered.length
  ).toFixed(2);

  const longestWord = filtered.reduce((a, b) => (b.length > a.length ? b : a));

  return { total: filtered.length, unique: freq.size, topWords, avgLen, longestWord };
}

// ── 主程序 ────────────────────────────────────────────────

const text = `Go is an open source programming language that makes it easy
              to build simple reliable and efficient software go go go`;

try {
  const result = analyzeText(text, { minLen: 3, topN: 3, exclude: ["and", "that"] });
  console.log(result);
  // { total: 13, unique: 11, topWords: [{word:'go',count:4},...], avgLen: 5.38, longestWord: 'programming' }
} catch (e) {
  if (e instanceof AnalysisError) console.error("Error:", e.message);
  else throw e;
}
```

---

### 三语言实现对照

| 代码段 | Go | Python | JavaScript |
|--------|----|--------|-----------|
| **返回结构体** | `struct` + `String()` 方法 | `@dataclass` + `__str__` | 对象字面量 `{}` |
| **可选参数** | Functional Options `func(*config)` | `*` 强制关键字 + 默认值 | 解构参数 `{ minLen=3 }={}` |
| **自定义错误** | `errors.New` / `fmt.Errorf` | `class X(Exception)` | `class X extends Error` |
| **闭包过滤器** | 匿名函数 `func(w string) bool` | `lambda w: ...` | 箭头函数 `(w) => ...` |
| **字符串清洗** | `strings.ToLower` + `strings.Trim` | `re.sub` + `str.lower` | `.replace(/re/g)` + `.toLowerCase` |
| **列表过滤** | `for range` + `append` | 列表推导 `[x for x if]` | `.filter()` |
| **词频统计** | `map[string]int` + `for range` | `Counter(list)` | `Map` + `.reduce()` |
| **多键排序** | `sort.Slice(i,j func() bool)` | `sorted(key=lambda x:(-c,w))` | `.sort(([wa,ca],[wb,cb])=>...)` |
| **平均值** | `for range` 累加 ÷ len | `sum(gen) / len` | `.reduce(sum) / length` |
| **最长词** | `for range` 逐个比较 | `max(lst, key=len)` | `.reduce((a,b)=>b.len>a.len?b:a)` |
| **空值安全** | `nil` + `if err != nil` | `or` 短路 / `if not x` | `?.` 可选链 + `??` 空值合并 |
