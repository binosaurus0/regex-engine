"""
NFA-Based Regex Engine
A complete regex implementation using graph traversal instead of backtracking
"""

import re
from typing import Dict, List, Tuple, Set, Optional, Union, Any

# Configuration
RE_REPEAT_LIMIT = 1000

class RegexEngine:
    """Main regex engine class"""
    
    def __init__(self):
        self.tokens = []
        self.pos = 0
    
    def match(self, pattern: str, text: str) -> bool:
        """Check if pattern matches the entire text"""
        try:
            ast = self.parse(pattern)
            return self.nfa_full_match(ast, text)
        except Exception as e:
            print(f"Error: {e}")
            return False
    
    def search(self, pattern: str, text: str) -> Optional[int]:
        """Find first occurrence of pattern in text"""
        try:
            ast = self.parse(pattern)
            for i in range(len(text)):
                if self.nfa_full_match(ast, text[i:]):
                    return i
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None

    # ==================== TOKENIZER ====================
    
    def tokenize(self, pattern: str) -> List[Tuple[str, str]]:
        """Convert pattern string into tokens"""
        tokens = []
        i = 0
        
        while i < len(pattern):
            ch = pattern[i]
            
            if ch == '\\' and i + 1 < len(pattern):
                # Escape sequences
                next_ch = pattern[i + 1]
                if next_ch in 'nrt\\.*+?{}[]()^$|':
                    if next_ch == 'n':
                        tokens.append(('CHAR', '\n'))
                    elif next_ch == 'r':
                        tokens.append(('CHAR', '\r'))
                    elif next_ch == 't':
                        tokens.append(('CHAR', '\t'))
                    else:
                        tokens.append(('CHAR', next_ch))
                    i += 2
                else:
                    tokens.append(('CHAR', next_ch))
                    i += 2
            elif ch == '.':
                tokens.append(('DOT', '.'))
                i += 1
            elif ch == '*':
                tokens.append(('STAR', '*'))
                i += 1
            elif ch == '+':
                tokens.append(('PLUS', '+'))
                i += 1
            elif ch == '?':
                tokens.append(('QUESTION', '?'))
                i += 1
            elif ch == '|':
                tokens.append(('PIPE', '|'))
                i += 1
            elif ch == '(':
                tokens.append(('LPAREN', '('))
                i += 1
            elif ch == ')':
                tokens.append(('RPAREN', ')'))
                i += 1
            elif ch == '{':
                # Handle quantifiers like {3}, {3,}, {3,5}
                j = i + 1
                while j < len(pattern) and pattern[j] != '}':
                    j += 1
                if j < len(pattern):
                    quantifier = pattern[i+1:j]
                    tokens.append(('QUANTIFIER', quantifier))
                    i = j + 1
                else:
                    tokens.append(('CHAR', ch))
                    i += 1
            elif ch == '[':
                # Character classes
                j = i + 1
                while j < len(pattern) and pattern[j] != ']':
                    if pattern[j] == '\\' and j + 1 < len(pattern):
                        j += 2
                    else:
                        j += 1
                if j < len(pattern):
                    char_class = pattern[i+1:j]
                    tokens.append(('CHARCLASS', char_class))
                    i = j + 1
                else:
                    tokens.append(('CHAR', ch))
                    i += 1
            elif ch == '^':
                tokens.append(('ANCHOR_START', '^'))
                i += 1
            elif ch == '$':
                tokens.append(('ANCHOR_END', '$'))
                i += 1
            else:
                tokens.append(('CHAR', ch))
                i += 1
        
        return tokens

    # ==================== PARSER ====================
    
    def parse(self, pattern: str):
        """Parse pattern into AST"""
        self.tokens = self.tokenize(pattern)
        self.pos = 0
        return self.parse_expression()
    
    def peek(self) -> Optional[Tuple[str, str]]:
        """Look at current token without consuming it"""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None
    
    def consume(self, expected_type: str = None) -> Optional[Tuple[str, str]]:
        """Consume and return current token"""
        if self.pos < len(self.tokens):
            token = self.tokens[self.pos]
            if expected_type is None or token[0] == expected_type:
                self.pos += 1
                return token
        return None
    
    def parse_expression(self):
        """Parse alternation (|)"""
        left = self.parse_concatenation()
        
        while self.peek() and self.peek()[0] == 'PIPE':
            self.consume('PIPE')
            right = self.parse_concatenation()
            left = ('split', left, right)
        
        return left
    
    def parse_concatenation(self):
        """Parse concatenation of terms"""
        terms = []
        
        while (self.peek() and 
               self.peek()[0] not in ['PIPE', 'RPAREN', 'ANCHOR_END']):
            terms.append(self.parse_term())
        
        if not terms:
            return None
        
        # Build left-associative concatenation tree
        result = terms[0]
        for term in terms[1:]:
            result = ('cat', result, term)
        
        return result
    
    def parse_term(self):
        """Parse a single term with optional quantifiers"""
        atom = self.parse_atom()
        
        # Handle quantifiers
        token = self.peek()
        if token:
            if token[0] == 'STAR':
                self.consume('STAR')
                return ('repeat', atom, 0, RE_REPEAT_LIMIT)
            elif token[0] == 'PLUS':
                self.consume('PLUS')
                return ('repeat', atom, 1, RE_REPEAT_LIMIT)
            elif token[0] == 'QUESTION':
                self.consume('QUESTION')
                return ('repeat', atom, 0, 1)
            elif token[0] == 'QUANTIFIER':
                self.consume('QUANTIFIER')
                rmin, rmax = self.parse_quantifier(token[1])
                return ('repeat', atom, rmin, rmax)
        
        return atom
    
    def parse_atom(self):
        """Parse atomic expressions"""
        token = self.peek()
        
        if not token:
            return None
        
        if token[0] == 'CHAR':
            self.consume('CHAR')
            return token[1]
        elif token[0] == 'DOT':
            self.consume('DOT')
            return 'dot'
        elif token[0] == 'LPAREN':
            self.consume('LPAREN')
            expr = self.parse_expression()
            self.consume('RPAREN')
            return expr
        elif token[0] == 'CHARCLASS':
            self.consume('CHARCLASS')
            return ('charclass', token[1])
        elif token[0] == 'ANCHOR_START':
            self.consume('ANCHOR_START')
            return ('anchor_start',)
        elif token[0] == 'ANCHOR_END':
            self.consume('ANCHOR_END')
            return ('anchor_end',)
        
        return None
    
    def parse_quantifier(self, quantifier: str) -> Tuple[int, int]:
        """Parse quantifier string like '3', '3,', '3,5'"""
        if ',' in quantifier:
            parts = quantifier.split(',', 1)
            rmin = int(parts[0]) if parts[0] else 0
            rmax = int(parts[1]) if parts[1] else RE_REPEAT_LIMIT
        else:
            rmin = rmax = int(quantifier)
        
        return rmin, min(rmax, RE_REPEAT_LIMIT)

    # ==================== NFA CONSTRUCTION ====================
    
    def nfa_make(self, node, start: List, end: List, id2node: Dict):
        """Build NFA graph from AST node"""
        if node is None:
            start.append((None, end))
        elif node == 'dot':
            start.append(('dot', end))
        elif isinstance(node, str):
            start.append((node, end))
        elif isinstance(node, tuple):
            if node[0] == 'cat':
                # Concatenation: connect via middle node
                middle = []
                id2node[id(middle)] = middle
                self.nfa_make(node[1], start, middle, id2node)
                self.nfa_make(node[2], middle, end, id2node)
            elif node[0] == 'split':
                # Alternation: connect to both branches
                self.nfa_make(node[1], start, end, id2node)
                self.nfa_make(node[2], start, end, id2node)
            elif node[0] == 'repeat':
                self.nfa_make_repeat(node, start, end, id2node)
            elif node[0] == 'charclass':
                start.append(('charclass', node[1], end))
            elif node[0] == 'anchor_start':
                start.append(('anchor_start', end))
            elif node[0] == 'anchor_end':
                start.append(('anchor_end', end))
        else:
            raise ValueError(f"Unknown node type: {node}")
    
    def nfa_make_repeat(self, node, start: List, end: List, id2node: Dict):
        """Create NFA structure for repetition"""
        _, subnode, rmin, rmax = node
        rmax = min(rmax, RE_REPEAT_LIMIT)
        
        # Door nodes for repetition control
        door_in = []
        door_out = ('boss', door_in, end, rmin, rmax)
        
        id2node[id(door_in)] = door_in
        id2node[id(door_out)] = door_out
        
        # Connect subgraph between doors
        self.nfa_make(subnode, door_in, door_out, id2node)
        
        # Connect start to door_in
        start.append((None, door_in))
        
        # If minimum is 0, also connect directly to end
        if rmin == 0:
            start.append((None, end))

    # ==================== NFA EXECUTION ====================
    
    def nfa_full_match(self, node, text: str) -> bool:
        """Execute NFA to match entire text"""
        # Build the graph
        start, end = [], []
        id2node = {id(start): start, id(end): end}
        self.nfa_make(node, start, end, id2node)
        
        # Initial position set
        node_set = {(id(start), ())}
        self.nfa_expand(node_set, id2node)
        
        # Process each character
        for i, ch in enumerate(text):
            node_set = self.nfa_step(node_set, ch, id2node, i, len(text))
            self.nfa_expand(node_set, id2node)
            
            # Early termination if no valid states
            if not node_set:
                return False
        
        # Check if we can reach the end
        self.nfa_expand_anchors(node_set, id2node, len(text), len(text))
        return (id(end), ()) in node_set
    
    def nfa_step(self, node_set: Set, ch: str, id2node: Dict, pos: int, text_len: int) -> Set:
        """Move to next position set by consuming input character"""
        next_nodes = set()
        
        for node_id, kv in node_set:
            node = id2node[node_id]
            
            # Skip boss nodes (handled by expand)
            if isinstance(node, tuple) and node[0] == 'boss':
                continue
            
            for link in node:
                if len(link) == 2:
                    cond, dst = link
                    if cond == 'dot' or cond == ch:
                        next_nodes.add((id(dst), kv))
                elif len(link) == 3 and link[0] == 'charclass':
                    _, char_class, dst = link
                    if self.match_char_class(ch, char_class):
                        next_nodes.add((id(dst), kv))
        
        return next_nodes
    
    def nfa_expand(self, node_set: Set, id2node: Dict):
        """Expand position set via free links and boss nodes"""
        to_process = list(node_set)
        
        while to_process:
            new_nodes = []
            
            for node_id, kv in to_process:
                node = id2node[node_id]
                
                if isinstance(node, tuple) and node[0] == 'boss':
                    # Handle boss node
                    node_set.discard((node_id, kv))
                    for dst, new_kv in self.nfa_boss(node, kv):
                        new_nodes.append((id(dst), new_kv))
                else:
                    # Handle regular node with free links
                    for link in node:
                        if len(link) == 2 and link[0] is None:
                            # Free link
                            _, dst = link
                            new_nodes.append((id(dst), kv))
            
            # Add new nodes and prepare for next iteration
            to_process = []
            for state in new_nodes:
                if state not in node_set:
                    node_set.add(state)
                    to_process.append(state)
    
    def nfa_expand_anchors(self, node_set: Set, id2node: Dict, pos: int, text_len: int):
        """Handle anchor expansions"""
        to_process = list(node_set)
        
        while to_process:
            new_nodes = []
            
            for node_id, kv in to_process:
                node = id2node[node_id]
                
                if not isinstance(node, tuple) or node[0] == 'boss':
                    continue
                
                for link in node:
                    if len(link) == 2:
                        cond, dst = link
                        if cond == 'anchor_start' and pos == 0:
                            new_nodes.append((id(dst), kv))
                        elif cond == 'anchor_end' and pos == text_len:
                            new_nodes.append((id(dst), kv))
            
            to_process = []
            for state in new_nodes:
                if state not in node_set:
                    node_set.add(state)
                    to_process.append(state)
    
    def nfa_boss(self, node, kv):
        """Handle boss node for repetition counting"""
        _, door_in, end, rmin, rmax = node
        key = id(door_in)
        kv, cnt = self.kv_increase(kv, key)
        
        if cnt < rmax:
            # Can repeat more
            yield (door_in, kv)
        
        if rmin <= cnt <= rmax:
            # Can exit
            yield (end, self.kv_delete(kv, key))
    
    # ==================== UTILITY FUNCTIONS ====================
    
    def kv_increase(self, kv, key):
        """Increment counter for key"""
        kv_dict = dict(kv)
        val = kv_dict.get(key, 0) + 1
        kv_dict[key] = val
        return tuple(sorted(kv_dict.items())), val
    
    def kv_delete(self, kv, key):
        """Remove key from counter set"""
        return tuple((k, v) for k, v in kv if k != key)
    
    def match_char_class(self, ch: str, char_class: str) -> bool:
        """Check if character matches character class"""
        if not char_class:
            return False
        
        negate = char_class.startswith('^')
        if negate:
            char_class = char_class[1:]
        
        matched = False
        i = 0
        
        while i < len(char_class):
            if i + 2 < len(char_class) and char_class[i + 1] == '-':
                # Range like a-z
                if char_class[i] <= ch <= char_class[i + 2]:
                    matched = True
                    break
                i += 3
            else:
                # Single character
                if ch == char_class[i]:
                    matched = True
                    break
                i += 1
        
        return matched != negate


# ==================== DEMO AND TESTS ====================

def test_regex_engine():
    """Test the regex engine with various patterns"""
    engine = RegexEngine()
    
    test_cases = [
        # Basic matching
        ("abc", "abc", True),
        ("abc", "ab", False),
        ("abc", "abcd", False),
        
        # Dot wildcard
        ("a.c", "abc", True),
        ("a.c", "axc", True),
        ("a.c", "ac", False),
        
        # Star quantifier
        ("ab*c", "ac", True),
        ("ab*c", "abc", True),
        ("ab*c", "abbbc", True),
        
        # Plus quantifier  
        ("ab+c", "ac", False),
        ("ab+c", "abc", True),
        ("ab+c", "abbbc", True),
        
        # Question mark
        ("ab?c", "ac", True),
        ("ab?c", "abc", True),
        ("ab?c", "abbc", False),
        
        # Alternation
        ("a|b", "a", True),
        ("a|b", "b", True),
        ("a|b", "c", False),
        
        # Groups
        ("(ab)+", "ab", True),
        ("(ab)+", "abab", True),
        ("(ab)+", "a", False),
        
        # Character classes
        ("[abc]", "a", True),
        ("[abc]", "b", True),
        ("[abc]", "d", False),
        ("[a-z]", "m", True),
        ("[^abc]", "d", True),
        ("[^abc]", "a", False),
        
        # Quantifiers
        ("a{3}", "aaa", True),
        ("a{3}", "aa", False),
        ("a{2,4}", "aa", True),
        ("a{2,4}", "aaa", True),
        ("a{2,4}", "aaaaa", False),
        
        # Complex patterns
        ("(a|b)*abb", "aabb", True),
        ("(a|b)*abb", "babb", True),
        ("(a|b)*abb", "ababb", True),
    ]
    
    print("Testing Regex Engine")
    print("=" * 50)
    
    passed = 0
    total = len(test_cases)
    
    for pattern, text, expected in test_cases:
        try:
            result = engine.match(pattern, text)
            status = "✓" if result == expected else "✗"
            print(f"{status} '{pattern}' vs '{text}' -> {result} (expected {expected})")
            if result == expected:
                passed += 1
        except Exception as e:
            print(f"✗ '{pattern}' vs '{text}' -> ERROR: {e}")
    
    print(f"\nPassed: {passed}/{total}")
    
    # Demo search functionality
    print("\nSearch Demo:")
    print("-" * 30)
    
    search_tests = [
        ("abc", "xyzabcdef", 3),
        ("a+", "bbaaaacc", 2), 
        ("\\d+", "abc123def", None),  # No digit support yet
    ]
    
    for pattern, text, expected in search_tests:
        result = engine.search(pattern, text)
        print(f"search('{pattern}', '{text}') -> {result} (expected {expected})")


if __name__ == "__main__":
    test_regex_engine()