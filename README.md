## **Architecture Overview**

**1. Tokenizer**: Converts regex patterns into tokens (characters, operators, quantifiers, etc.)

**2. Parser**: Builds an Abstract Syntax Tree (AST) from tokens, handling:
- Concatenation and alternation
- Groups with parentheses
- Quantifiers (`*`, `+`, `?`, `{n,m}`)
- Character classes `[abc]`, ranges `[a-z]`, negation `[^abc]`
- Anchors `^` and `$`
- Dot wildcard `.`

**3. NFA Constructor**: Converts AST into a graph representation with:
- Regular nodes connected by character conditions
- "Boss" nodes for repetition counting
- Free links (epsilon transitions)

**4. NFA Executor**: Traverses the graph using breadth-first search:
- Maintains sets of possible positions
- Handles repetition counters
- Expands through free links

## **Key Advantages Over Backtracking**

1. **No exponential blowup** - processes each input character once
2. **Breadth-first search** instead of depth-first
3. **Parallel state tracking** - maintains all possible matches simultaneously
4. **Better performance** on complex patterns with nested quantifiers

## **Supported Features**

- ✅ Basic matching: `abc`
- ✅ Wildcards: `a.c`
- ✅ Quantifiers: `*`, `+`, `?`, `{n,m}`
- ✅ Alternation: `a|b`
- ✅ Groups: `(ab)+`
- ✅ Character classes: `[abc]`, `[a-z]`, `[^abc]`
- ✅ Anchors: `^start`, `end$`
- ✅ Escape sequences: `\n`, `\t`, `\\`

The engine includes comprehensive tests and demonstrates both `match()` (full string) and `search()` (find substring) functionality. The NFA approach makes it much more efficient than backtracking-based regex engines, especially for patterns with complex quantifiers and alternations.