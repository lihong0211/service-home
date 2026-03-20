# 前端面试题精要

---

## 一、JavaScript

### 1. var / let / const 的区别

| | var | let | const |
|---|---|---|---|
| 作用域 | 函数 | 块级 | 块级 |
| 变量提升 | 是（undefined）| 是（TDZ，不可用）| 是（TDZ，不可用）|
| 重复声明 | 允许 | 不允许 | 不允许 |
| 可重赋值 | 是 | 是 | 否 |

> TDZ（暂时性死区）：声明前访问会抛 `ReferenceError`。

---

### 2. 闭包

**定义：** 函数 + 其能访问的外部词法作用域。  
**本质：** 内层函数引用了外层函数的变量，导致外层作用域不被回收。

```js
function counter() {
  let count = 0
  return () => ++count
}
const inc = counter()
inc() // 1
inc() // 2
```

**用途：** 数据私有化、函数工厂、防抖/节流。  
**副作用：** 变量常驻内存，不当使用造成内存泄漏。

---

### 3. 原型链

- 每个对象有 `__proto__` 指向其构造函数的 `prototype`。
- 访问属性时沿链向上查找，直到 `null` 为止。
- `Object.prototype.__proto__ === null`（链顶端）。

```js
function Dog() {}
const d = new Dog()
d.__proto__ === Dog.prototype        // true
Dog.prototype.__proto__ === Object.prototype  // true
```

---

### 4. 事件循环（Event Loop）

**执行顺序：** 同步代码 → 微任务队列 → 宏任务队列（循环）

| 类型 | 示例 |
|---|---|
| 微任务 | `Promise.then`、`queueMicrotask`、`MutationObserver` |
| 宏任务 | `setTimeout`、`setInterval`、`I/O`、`requestAnimationFrame` |

> 每次宏任务执行完后，立即清空所有微任务，再执行下一个宏任务。

---

### 5. Promise / async-await

```js
// Promise 链式
fetch(url).then(res => res.json()).then(data => ...).catch(err => ...)

// async/await（语法糖，底层仍是 Promise）
async function load() {
  try {
    const res = await fetch(url)
    const data = await res.json()
  } catch(e) { ... }
}
```

**Promise 状态：** `pending` → `fulfilled` / `rejected`（不可逆）  
**常用方法：**
- `Promise.all`：全部成功才成功，一个失败即失败
- `Promise.allSettled`：全部结束，不论成功失败
- `Promise.race`：第一个结束的结果
- `Promise.any`：第一个成功的结果

---

### 6. this 指向

| 调用方式 | this |
|---|---|
| 普通函数调用 | `window`（严格模式 `undefined`）|
| 方法调用 | 调用该方法的对象 |
| `new` 调用 | 新创建的实例 |
| `call/apply/bind` | 指定的第一个参数 |
| 箭头函数 | 定义时的外层 `this`（不可更改）|

---

### 7. 深拷贝 vs 浅拷贝

- **浅拷贝：** 只复制第一层，引用类型仍共享。`Object.assign`、展开运算符 `{...obj}`。
- **深拷贝：** 递归复制所有层。

```js
// 简单场景
const deep = JSON.parse(JSON.stringify(obj))  // 不支持函数/undefined/循环引用

// 通用方案
const deep = structuredClone(obj)  // 现代浏览器原生支持
```

---

### 8. 防抖（debounce）vs 节流（throttle）

| | 防抖 | 节流 |
|---|---|---|
| 触发时机 | 停止触发 N 毫秒后执行一次 | 每隔 N 毫秒执行一次 |
| 场景 | 搜索输入框、窗口 resize | 滚动事件、按钮连点 |

```js
// 防抖
function debounce(fn, delay) {
  let timer
  return (...args) => {
    clearTimeout(timer)
    timer = setTimeout(() => fn(...args), delay)
  }
}

// 节流
function throttle(fn, interval) {
  let last = 0
  return (...args) => {
    const now = Date.now()
    if (now - last >= interval) { last = now; fn(...args) }
  }
}
```

---

### 9. 箭头函数 vs 普通函数

| | 箭头函数 | 普通函数 |
|---|---|---|
| `this` | 继承外层（词法） | 动态绑定 |
| `arguments` | 无 | 有 |
| `new` | 不可用 | 可用 |
| `prototype` | 无 | 有 |

---

### 10. == vs ===

- `===` 严格相等：类型和值都相同。
- `==` 宽松相等：会做**类型转换**（隐患大，尽量避免）。

```js
0 == false   // true（false 转 0）
null == undefined  // true（特例）
null === undefined  // false
```

---

### 11. typeof vs instanceof

```js
typeof 42         // "number"
typeof null       // "object"（历史 bug）
typeof function(){} // "function"

[] instanceof Array    // true
[] instanceof Object   // true（万物皆对象）
```

> 判断具体类型用：`Object.prototype.toString.call(val)` → `"[object Array]"`

---

### 12. 垃圾回收

- **标记清除（主流）：** 从根对象出发，标记所有可达对象，未被标记的回收。
- **引用计数（旧）：** 引用次数为 0 时回收，但有循环引用问题。
- **WeakMap / WeakRef：** 弱引用，不阻止 GC。

---

### 13. 作用域链

函数查找变量时，先找自身作用域 → 外层函数作用域 → ... → 全局，找不到抛 `ReferenceError`。

---

### 14. 手写 new

```js
function myNew(Constructor, ...args) {
  const obj = Object.create(Constructor.prototype)  // 创建实例，设置原型
  const result = Constructor.apply(obj, args)        // 执行构造函数
  return result instanceof Object ? result : obj     // 若构造函数返回对象则用它
}
```

---

### 15. 手写 call / apply / bind

```js
Function.prototype.myCall = function(ctx, ...args) {
  ctx = ctx || window
  const sym = Symbol()
  ctx[sym] = this
  const res = ctx[sym](...args)
  delete ctx[sym]
  return res
}

Function.prototype.myBind = function(ctx, ...args) {
  return (...rest) => this.call(ctx, ...args, ...rest)
}
```

---

### 16. 数组常用方法

| 方法 | 说明 | 是否改变原数组 |
|---|---|---|
| `push/pop` | 末尾增/删 | 是 |
| `shift/unshift` | 头部删/增 | 是 |
| `splice(i, n, ...items)` | 删除/插入 | 是 |
| `slice(start, end)` | 截取 | 否 |
| `map/filter/reduce` | 遍历转换 | 否 |
| `find/findIndex` | 查找 | 否 |
| `flat(depth)` | 展平 | 否 |
| `sort` | 排序 | 是 |

---

### 17. 柯里化（Currying）

将多参函数转为逐步传参的单参函数。

```js
const curry = (fn) => {
  const arity = fn.length
  return function curried(...args) {
    if (args.length >= arity) return fn(...args)
    return (...more) => curried(...args, ...more)
  }
}
```

---

### 18. 模块化：ESM vs CJS

| | ESM (`import/export`) | CJS (`require/module.exports`) |
|---|---|---|
| 加载时机 | 编译时静态分析 | 运行时动态 |
| `this` | `undefined` | `module` |
| 异步 | 支持 `import()` | 不支持 |
| Tree Shaking | 支持 | 不支持 |

---

### 19. Map vs WeakMap

| | Map | WeakMap |
|---|---|---|
| 键类型 | 任意 | 只能是对象 |
| 是否可迭代 | 是 | 否 |
| GC 影响 | 阻止回收 | 不阻止（弱引用）|

---

### 20. 事件委托

利用事件冒泡，把子元素的事件监听绑定在父元素上：
- 减少监听器数量，提升性能
- 动态新增子元素无需重新绑定

```js
document.getElementById('list').addEventListener('click', (e) => {
  if (e.target.tagName === 'LI') console.log(e.target.textContent)
})
```

---

---

## 二、Vue

### 1. Vue2 vs Vue3 核心差异

| | Vue2 | Vue3 |
|---|---|---|
| 响应式 | `Object.defineProperty` | `Proxy` |
| API 风格 | Options API | Composition API（`setup`）|
| 性能 | — | 更快（编译优化、Tree Shaking）|
| TypeScript | 弱支持 | 原生支持 |
| 根节点 | 单根 | 支持多根（Fragment）|

---

### 2. 响应式原理

**Vue2：** `Object.defineProperty` 劫持 getter/setter，缺点：无法检测新增/删除属性、数组索引变更（需 `$set`）。

**Vue3：** `Proxy` 代理整个对象，可拦截所有操作，支持新增属性和数组下标变更，性能更好。

---

### 3. v-if vs v-show

| | v-if | v-show |
|---|---|---|
| 原理 | 销毁/重建 DOM | `display: none/block` |
| 首次渲染 | 惰性（条件为假不渲染）| 总是渲染 |
| 切换开销 | 高 | 低 |
| 适用场景 | 条件很少改变 | 频繁切换显示 |

---

### 4. computed vs watch

| | computed | watch |
|---|---|---|
| 用途 | 派生值（有返回值）| 监听值变化，执行副作用 |
| 缓存 | 是（依赖不变不重算）| 否 |
| 异步 | 不支持 | 支持 |
| 场景 | 模板中的计算属性 | 数据变化后请求接口、操作 DOM |

---

### 5. 生命周期（Vue3 Composition API）

```
setup()
  ↓
onBeforeMount → onMounted          ← 挂载
  ↓
onBeforeUpdate → onUpdated         ← 更新
  ↓
onBeforeUnmount → onUnmounted      ← 卸载
```

> Vue2 对应：`beforeCreate/created` → `setup()`；`beforeDestroy/destroyed` → `onBeforeUnmount/onUnmounted`

---

### 6. 组件通信方式

| 方式 | 方向 |
|---|---|
| `props` / `emits` | 父 → 子 / 子 → 父 |
| `v-model` | 父子双向 |
| `provide` / `inject` | 跨层级祖先 → 后代 |
| `Pinia / Vuex` | 全局状态 |
| `EventBus` / `mitt` | 任意组件（兄弟等）|
| `$refs` | 父访问子实例 |

---

### 7. ref vs reactive

```js
const count = ref(0)       // 基本类型，访问需 .value
const state = reactive({}) // 对象/数组，直接访问属性
```

- `ref` 内部对对象类型也会用 `reactive` 包装。
- 解构 `reactive` 会失去响应性，用 `toRefs()` 解决。

---

### 8. Pinia vs Vuex

| | Pinia | Vuex |
|---|---|---|
| API | Composition 风格，简洁 | Options 风格，有 mutation |
| TypeScript | 原生支持 | 需额外配置 |
| Devtools | 支持 | 支持 |
| 模块化 | 天然扁平（每个 store 独立）| 需 `modules` |
| Vue3 推荐 | ✅ 官方推荐 | 维护模式 |

---

### 9. Vue Router 导航守卫

```
全局前置守卫 beforeEach
  → 路由独享守卫 beforeEnter
    → 组件内守卫 beforeRouteEnter
      → 全局解析守卫 beforeResolve
        → 全局后置钩子 afterEach（无 next）
```

---

### 10. keep-alive

缓存组件实例，避免重复挂载/销毁。被缓存的组件触发 `onActivated` / `onDeactivated` 钩子（而非 mounted/unmounted）。

```html
<keep-alive include="A,B" :max="10">
  <component :is="current" />
</keep-alive>
```

---

### 11. 为什么 v-for 需要 key

Diff 算法通过 `key` 识别节点身份，实现节点复用而非销毁重建，提升性能。  
用 `index` 作 key 在列表重排时会出错（复用错误节点），应用唯一 ID。

---

### 12. nextTick

DOM 更新是异步的（批量），`nextTick` 在下次 DOM 更新后执行回调，用于获取更新后的 DOM。

```js
state.count++
await nextTick()
console.log(el.textContent) // 已更新
```

---

### 13. Diff 算法（Vue3）

Vue3 使用**最长递增子序列（LIS）**算法优化节点移动，减少 DOM 操作次数。  
核心策略：双端比较 → 新旧头尾四指针 → 遍历中间乱序节点 → 按 LIS 最小移动。

---

### 14. 自定义指令

```js
app.directive('focus', {
  mounted(el) { el.focus() }
})
```

钩子：`created` → `beforeMount` → `mounted` → `beforeUpdate` → `updated` → `beforeUnmount` → `unmounted`

---

---

## 三、React

### 1. Class 组件 vs 函数组件

| | Class 组件 | 函数组件 |
|---|---|---|
| 状态 | `this.state` | `useState` |
| 生命周期 | 钩子方法 | `useEffect` |
| 性能 | 相对重 | 轻量 |
| 代码复用 | HOC、render props | 自定义 Hooks |
| 趋势 | 逐渐淘汰 | ✅ 主流 |

---

### 2. useState

```js
const [count, setCount] = useState(0)
setCount(prev => prev + 1)  // 函数式更新，避免闭包陷阱
```

- 状态更新是**异步合并**的（React 18 自动批处理）。
- 不能直接修改 state，必须用 setter。

---

### 3. useEffect

```js
useEffect(() => {
  // 副作用：请求、订阅、DOM 操作
  return () => { /* 清理函数 */ }
}, [deps])  // 依赖数组
```

| 依赖数组 | 执行时机 |
|---|---|
| 不传 | 每次渲染后 |
| `[]` | 仅挂载后执行一次 |
| `[a, b]` | `a` 或 `b` 变化后执行 |

---

### 4. useMemo vs useCallback

```js
const value = useMemo(() => expensiveCalc(a, b), [a, b])     // 缓存计算结果
const handler = useCallback(() => doSomething(id), [id])     // 缓存函数引用
```

> 本质都是缓存，避免子组件不必要的重渲染（配合 `React.memo` 使用）。

---

### 5. 虚拟 DOM（Virtual DOM）

用 JS 对象描述真实 DOM 结构，每次更新先比较新旧虚拟 DOM（Diff），再将差异最小化地应用到真实 DOM，减少昂贵的 DOM 操作。

**优势：** 跨平台、批量更新、开发效率高。  
**并非"更快"**，直接操作 DOM 在简单场景更快，虚拟 DOM 的价值在于**可维护性和足够好的性能**。

---

### 6. React Fiber

React 16 重写的协调引擎，核心目标：**可中断渲染**。

- 将渲染任务拆分为小的 Fiber 节点（链表结构）
- 利用 `requestIdleCallback`（类似）在空闲时间处理
- 支持优先级调度（高优先级任务插队）
- 支持并发模式（Concurrent Mode）

---

### 7. 受控 vs 非受控组件

| | 受控组件 | 非受控组件 |
|---|---|---|
| 数据存储 | React state | DOM 自身 |
| 获取值 | `value` + `onChange` | `ref.current.value` |
| 场景 | 需要即时验证、联动 | 简单表单、文件上传 |

---

### 8. React.memo / PureComponent

```js
const Child = React.memo(({ count }) => <div>{count}</div>)
// 若 props 未变化，跳过重渲染
```

- `React.memo`：函数组件的浅比较 props
- `PureComponent`：类组件的浅比较 props + state
- 深层对象需结合 `useMemo` / `useCallback`

---

### 9. Context

```js
const ThemeCtx = createContext('light')

// 提供
<ThemeCtx.Provider value="dark">
  <App />
</ThemeCtx.Provider>

// 消费
const theme = useContext(ThemeCtx)
```

> 值变化时，所有消费该 Context 的组件都会重渲染，需谨慎拆分 Context。

---

### 10. Hooks 使用规则

1. **只在函数组件或自定义 Hook 顶层调用**（不能在循环、条件、嵌套函数中调用）
2. **只在 React 函数组件或自定义 Hook 中调用**

原因：React 依赖调用**顺序**来对应每次渲染的 Hook 状态，条件/循环会打乱顺序。

---

### 11. 自定义 Hook

```js
function useFetch(url) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  useEffect(() => {
    fetch(url).then(r => r.json()).then(d => { setData(d); setLoading(false) })
  }, [url])
  return { data, loading }
}
```

---

### 12. 合成事件（SyntheticEvent）

React 将原生事件封装为合成事件，统一浏览器差异，事件委托到根节点（React 17+ 改为 root，之前是 document）。  
合成事件对象会被**复用**（React 16 事件池，17+ 移除），异步使用需 `e.persist()`（16）或直接赋值。

---

### 13. Redux 核心概念

- **Store：** 全局唯一状态树
- **Action：** 描述变更的普通对象 `{ type, payload }`
- **Reducer：** 纯函数，`(state, action) => newState`
- **Dispatch：** 触发 Action
- **Selector：** 从 Store 读取数据

**数据流：** UI dispatch Action → Reducer 计算新 State → Store 更新 → UI 重渲染

---

### 14. useReducer vs useState

```js
const [state, dispatch] = useReducer(reducer, initialState)
```

- 状态逻辑复杂（多个子值互相影响）时用 `useReducer`
- 简单计数、toggle 用 `useState`

---

### 15. React 性能优化

1. `React.memo` / `PureComponent` 防止不必要重渲染
2. `useMemo` / `useCallback` 缓存值和函数
3. 虚拟列表（`react-window`）处理大数据列表
4. 代码分割 `React.lazy` + `Suspense`
5. 避免匿名函数/对象作为 props
6. 合理拆分 Context，避免大范围重渲染

---

---

## 四、浏览器 & 网络基础

### 1. 浏览器渲染流程

```
HTML 解析 → DOM 树
CSS 解析 → CSSOM 树
        ↓
    合并 Render Tree
        ↓
    Layout（布局/回流）— 计算位置尺寸
        ↓
    Paint（绘制）— 生成像素
        ↓
    Composite（合成）— 图层合并显示
```

---

### 2. 重排（Reflow）vs 重绘（Repaint）

| | 重排 | 重绘 |
|---|---|---|
| 触发 | 几何属性变化（宽高、位置）| 视觉属性变化（颜色、背景）|
| 开销 | 大（重新布局）| 小（仅重绘）|
| 关系 | 必然触发重绘 | 不一定触发重排 |

**减少重排：** 批量修改样式（`classList`）、用 `transform/opacity` 做动画（GPU 合成层）、避免频繁读取 `offsetWidth` 等布局属性。

---

### 3. HTTP vs HTTPS

- HTTPS = HTTP + TLS/SSL 加密层
- 作用：数据加密（防窃听）、身份验证（防伪造）、数据完整性（防篡改）
- TLS 握手：证书验证 → 密钥交换 → 对称加密通信

---

### 4. HTTP 状态码

| 范围 | 含义 | 常见 |
|---|---|---|
| 2xx | 成功 | 200 OK、201 Created、204 No Content |
| 3xx | 重定向 | 301 永久、302 临时、304 Not Modified |
| 4xx | 客户端错误 | 400 Bad Request、401 未授权、403 禁止、404 Not Found |
| 5xx | 服务端错误 | 500 Internal Error、502 Bad Gateway、503 不可用 |

---

### 5. 跨域（CORS）

**同源策略：** 协议 + 域名 + 端口三者相同才算同源。

**解决方案：**
- **CORS（主流）：** 服务端设置响应头 `Access-Control-Allow-Origin`
- **反向代理：** 开发时 webpack/vite 代理，生产时 nginx 转发
- **JSONP：** 只支持 GET，利用 `<script>` 标签不受同源限制（已过时）

---

### 6. 浏览器缓存

```
强缓存（不请求服务器）
  Cache-Control: max-age=3600
  Expires: xxx（旧）

协商缓存（请求服务器，服务器判断是否更新）
  ETag / If-None-Match（内容 hash）
  Last-Modified / If-Modified-Since（修改时间）
  → 未变化：304 Not Modified（使用缓存）
  → 已变化：200 + 新内容
```

---

### 7. XSS vs CSRF

| | XSS（跨站脚本）| CSRF（跨站请求伪造）|
|---|---|---|
| 原理 | 注入恶意脚本在受害者浏览器执行 | 诱导受害者发送已认证的请求 |
| 防御 | 转义输出、CSP、HttpOnly Cookie | CSRF Token、SameSite Cookie、验证 Referer |

---

### 8. CSS 盒模型

```
标准盒模型：width = content（不含 padding/border）
IE 盒模型：width = content + padding + border

box-sizing: content-box  // 标准（默认）
box-sizing: border-box   // IE 模型（推荐，更直观）
```

---

### 9. Flex vs Grid

| | Flexbox | Grid |
|---|---|---|
| 维度 | 一维（行或列）| 二维（行和列）|
| 场景 | 导航栏、按钮组、单行/列布局 | 整体页面布局、卡片网格 |
| 对齐 | `justify-content`、`align-items` | `justify-items`、`align-items` + 区域定义 |

---

### 10. 前端性能优化

**加载优化：**
- 资源压缩（gzip/brotli）、代码分割、懒加载、CDN
- 减少 HTTP 请求（合并资源、雪碧图）
- 图片优化（WebP、适当尺寸、懒加载）
- 预加载关键资源 `<link rel="preload">`

**运行时优化：**
- 减少重排重绘（transform/opacity 动画）
- 虚拟滚动（大列表）
- Web Worker（CPU 密集任务移出主线程）
- 防抖/节流高频事件

---

### 11. localStorage vs sessionStorage vs Cookie

| | localStorage | sessionStorage | Cookie |
|---|---|---|---|
| 大小 | ~5MB | ~5MB | ~4KB |
| 生命周期 | 永久（手动清除）| 标签关闭清除 | 设置过期时间 |
| 随请求发送 | 否 | 否 | 是 |
| 作用域 | 同源 | 同源+同标签页 | 可设置域/路径 |

---

### 12. 输入 URL 到页面呈现（完整流程）

1. **DNS 解析**：域名 → IP
2. **TCP 三次握手**：建立连接
3. **TLS 握手**：HTTPS 加密（如有）
4. **发送 HTTP 请求**
5. **服务器响应**，浏览器接收 HTML
6. **解析 HTML → DOM 树**，下载 CSS/JS
7. **解析 CSS → CSSOM**，合并为渲染树
8. **Layout → Paint → Composite**：显示页面
9. **执行 JS**，DOM/CSSOM 可能更新

---

*文档共计 60+ 题，持续更新。*
