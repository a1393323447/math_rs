use std::collections::{HashMap, HashSet};
use std::ops::{BitAnd, BitOr, RangeInclusive};

#[derive(Debug, Clone, Copy)]
pub enum Condition {
    Epsilon,
    Symbol(char),
}

impl Condition {
    fn is_epsilon_condition(&self) -> bool {
        match self {
            Condition::Epsilon => true,
            Condition::Symbol(_) => false,
        }
    }
    fn is_match(&self, symbol: char) -> bool {
        match self {
            Condition::Epsilon => false,
            Condition::Symbol(s) => *s == symbol,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NFAEdge(pub u32, pub u32, pub Condition);

// TODO 使用前缀数组对 states 的检索进行优化 （可能没什么用）
// TODO 求 e_closure 的过程本质是一个深度历遍, 考虑用栈优化
#[derive(Debug, Clone)]
pub struct NFA<AcceptEnum> {
    edges: Vec<NFAEdge>,
    vertex_num: u32,
    states: HashMap<u32, AcceptEnum>,
    symbol_set: HashSet<char>,
}

impl<Ac> NFA<Ac> {
    pub fn new(
        edges: Vec<NFAEdge>,
        vertex_num: u32,
        states: HashMap<u32, Ac>,
        symbol_set: HashSet<char>,
    ) -> NFA<Ac> {
        NFA { edges, vertex_num, states, symbol_set }
    }
    pub fn zero_or_one(symbol: char) -> NFA<Ac> {
        let edges = vec![NFAEdge(0, 1, Condition::Symbol(symbol)), NFAEdge(0, 1, Condition::Epsilon)];
        let mut symbol_set = HashSet::new();
        symbol_set.insert(symbol);
        NFA { edges, vertex_num: 2, states: HashMap::new(), symbol_set }
    }
    pub fn from_symbol(symbol: char) -> NFA<Ac> {
        let edges = vec![NFAEdge(0, 1, Condition::Symbol(symbol))];
        let mut symbol_set = HashSet::new();
        symbol_set.insert(symbol);
        NFA { edges, vertex_num: 2, states: HashMap::new(), symbol_set }
    }
    pub fn from_literal(s: &str) -> NFA<Ac> {
        let mut edges = Vec::new();
        let mut symbol_set = HashSet::new();
        let mut vertex_num = 0;
        for symbol in s.chars() {
            edges.push(NFAEdge(vertex_num, vertex_num+1, Condition::Symbol(symbol)));
            symbol_set.insert(symbol);
            vertex_num += 1;
        }
        vertex_num += 1;
        NFA { edges, vertex_num, states: HashMap::new(), symbol_set }
    }
    pub fn from_symbol_set(s: &str) -> NFA<Ac> {
        let mut edges = Vec::new();
        let mut symbol_set = HashSet::new();
        let mut vertex_num = 0;
        for symbol in s.chars() {
            vertex_num += 1;
            edges.push(NFAEdge(0, vertex_num, Condition::Symbol(symbol)));
            symbol_set.insert(symbol);
        }
        vertex_num += 1;
        for vertex in 1..vertex_num {
            edges.push(NFAEdge(vertex, vertex_num, Condition::Epsilon));
        }
        vertex_num += 1;
        NFA { edges, vertex_num, states: HashMap::new(), symbol_set }
    }
    pub fn from_symbol_range(symbol_rng: RangeInclusive<char>) -> NFA<Ac> {
        let mut edges = Vec::new();
        let mut symbol_set = HashSet::new();
        let mut vertex_num = 0;
        for symbol in symbol_rng {
            vertex_num += 1;
            edges.push(NFAEdge(0, vertex_num, Condition::Symbol(symbol)));
            symbol_set.insert(symbol);
        }
        vertex_num += 1;
        for vertex in 1..vertex_num {
            edges.push(NFAEdge(vertex, vertex_num, Condition::Epsilon));
        }
        vertex_num += 1;
        NFA { edges, vertex_num, states: HashMap::new(), symbol_set }
    }
    pub fn set_state(&mut self, state: Ac) {
        assert!(self.states.len() <= 1);
        if self.states.is_empty() {
            self.states.insert(self.vertex_num - 1, state);
        } else {
            *self.states.get_mut(&(self.vertex_num - 1)).unwrap() = state;
        }
    }
    pub fn e_closure_with_vertex(&self, vertex: u32) -> HashSet<u32> {
        assert!(vertex < self.vertex_num);
        let mut nfa_vertexs = HashSet::new();
        nfa_vertexs.insert(vertex);
        self.e_closure(&nfa_vertexs)
    }
    pub fn e_closure(&self, vertexs: &HashSet<u32>) -> HashSet<u32> {
        let mut result = HashSet::new();
        let mut stack = vec![];
        for vertex in vertexs {
            stack.push(*vertex);
            while !stack.is_empty() {
                let current_vertex = stack.pop().unwrap();
                result.insert(current_vertex);
                for NFAEdge(start, end, condition) in &self.edges {
                    if *start == current_vertex && condition.is_epsilon_condition() {
                        if !result.contains(end) {
                            stack.push(*end);
                        }
                        result.insert(*end);
                    } else if *start > current_vertex {
                        break;
                    }
                }
            }
        }
        result
    }
    pub fn move_set(&self, nfa_vertexs: &HashSet<u32>, symbol: char) -> HashSet<u32> {
        let mut result = HashSet::new();
        for vertex in nfa_vertexs {
            for NFAEdge(start, end, condition) in &self.edges {
                if *start == *vertex && condition.is_match(symbol) {
                    result.insert(*end);
                    let mut set = HashSet::new();
                    set.insert(*end);
                    let e_closure = self.e_closure(&set);
                    result = &result | &e_closure
                } else if *start > *vertex {
                    break;
                }
            }
        }
        result
    }
    pub fn get_states(&self) -> &HashMap<u32, Ac> {
        &self.states
    }
    pub fn get_vertex_num(&self) -> u32 {
        self.vertex_num
    }
    pub fn symbol_set(&self) -> &HashSet<char> {
        &self.symbol_set
    }
    pub fn get_edges(&self) -> &Vec<NFAEdge> {
        &self.edges
    }
}

impl<Ac: Copy> NFA<Ac> {
    pub fn get_state(&self, vertexs: &HashSet<u32>) -> Option<Ac> {
        for v in vertexs {
            match self.states.get(v) {
                Some(ac) => { return Some(*ac); },
                None => { continue; },
            }
        }
        None
    }
}

impl<Ac: Copy> BitOr for NFA<Ac>  {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        let tvn = self.vertex_num;
        let ovn = rhs.vertex_num;

        let new_start = 0;
        let new_end = tvn + ovn + 1;
        let new_vertex_num = tvn + ovn + 2;
        let new_symbol_set = &self.symbol_set | &rhs.symbol_set;
        let new_states = {
            let mut result = HashMap::new();
            self.states.iter().for_each(|(_, state)| { result.insert(new_end, *state); });
            rhs.states.iter().for_each(|(_, state)| { result.insert(new_end, *state); });
            result
        };

        let new_edges = {
            let mut edges = Vec::new();
            //              -> self.start(1)         First
            // new_start(0)
            //              -> rhs.start(tvn + 1)    Second
            edges.push(NFAEdge(new_start, 1, Condition::Epsilon));     // First
            edges.push(NFAEdge(new_start, tvn+1, Condition::Epsilon)); // Second
            // 
            // self.start(0 + 1) -> ... (n + 1) -> self.end(tvn) -> new_end(ovn + tvn + 1)
            //
            for NFAEdge(start, end, condition) in self.edges {
                edges.push(NFAEdge(start + 1, end + 1, condition));
            }
            edges.push(NFAEdge(tvn, new_end, Condition::Epsilon));
            //
            // rhs.start(0 + tvn + 1) -> ... (n + tvn + 1) -> rhs.end(ovn + tvn) -> new_end(ovn + tvn + 1)
            //
            for NFAEdge(start, end, condition) in rhs.edges {
                edges.push(NFAEdge(start + tvn + 1, end + tvn + 1, condition));
            }
            edges.push(NFAEdge(ovn + tvn, new_end, Condition::Epsilon));
            // 注意到现在的 edges 中的元素 (start, ..) 是按照以 start 作为升序排列
            edges
        };
        NFA { edges: new_edges, vertex_num: new_vertex_num, states: new_states, symbol_set: new_symbol_set }
    }
}

impl<Ac: Copy> BitAnd for NFA<Ac> {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        let tvn = self.vertex_num;
        let ovn = rhs.vertex_num;

        let new_start = 0;
        let new_end = tvn + ovn + 1;
        let new_vertex_num = tvn + ovn + 2;
        let new_symbol_set = &self.symbol_set | &rhs.symbol_set;
        let new_states = {
            assert!(self.states.len() + rhs.states.len() <= 1);
            let mut result = HashMap::new();
            self.states.iter().for_each(|(_, state)| { result.insert(new_end, *state); });
            rhs.states.iter().for_each(|(_, state)| { result.insert(new_end, *state); });
            result
        };
        let new_edges = {
            let mut edges = Vec::new();
            // 
            // new_start(0) -> self.start(0 + 1) -> ... (n + 1) -> self.end(tvn)
            //
            edges.push(NFAEdge(new_start, 1, Condition::Epsilon)); // new_start(0) -> self.start(0 + 1)
            // self.start(0 + 1) -> ... (n + 1) -> self.end(tvn)
            for NFAEdge(start, end, condition) in self.edges {
                edges.push(NFAEdge(start + 1, end + 1, condition));
            }
            //
            // self.end(tvn) -> rhs.start(0 + tvn + 1) -> ... (n + tvn + 1) -> rhs.end(ovn + tvn) -> new_end(ovn + tvn + 1)
            //
            edges.push(NFAEdge(tvn, tvn + 1, Condition::Epsilon)); // self.end(tvn) -> rhs.start(0 + tvn + 1)
            // rhs.start(0 + tvn + 1) -> ... (n + tvn + 1) -> rhs.end(ovn + tvn) -> new_end(ovn + tvn + 1)
            for NFAEdge(start, end, condition) in rhs.edges {
                edges.push(NFAEdge(start + tvn + 1, end + tvn + 1, condition));
            }
            edges.push(NFAEdge(ovn + tvn, new_end, Condition::Epsilon));
            // 注意到现在的 edges 中的元素 (start, ..) 是按照以 start 作为升序排列
            edges
        };
        NFA { edges: new_edges, vertex_num: new_vertex_num, states: new_states, symbol_set: new_symbol_set }
    }
}

impl<Ac> NFA<Ac> {
    pub fn closure(mut self) -> NFA<Ac> {
        self.edges.insert(0, NFAEdge(0, self.vertex_num - 1, Condition::Epsilon));
        self.edges.push(NFAEdge(self.vertex_num - 1, 0, Condition::Epsilon));
        self
    }
}

#[macro_export]
macro_rules! link_nfa {
    ($($nfa: expr), *) => {
        {
            use std::collections::{HashMap, HashSet};
            use crate::tokenizer::nfa::*;
            let mut vertex_num = 1;
            let mut edges = Vec::new();
            let mut symbol_set = HashSet::new();
            let mut states = HashMap::new();
            let mut vertex_num_epsilon = 1;
            $(
                edges.push(NFAEdge(0, vertex_num_epsilon, Condition::Epsilon));
                #[allow(unused)]
                vertex_num_epsilon += $nfa.get_vertex_num();
            )*
            $(
                let current_vertex = vertex_num;
                for NFAEdge(start, end, condition) in $nfa.get_edges() {
                    edges.push(NFAEdge(*start + current_vertex, *end + current_vertex, *condition));
                }
                vertex_num += $nfa.get_vertex_num();
                symbol_set = &symbol_set | $nfa.symbol_set();
                $nfa.get_states().iter().for_each(|(v, s)| { states.insert(v + current_vertex, *s); } );
            )*
            NFA::new(edges, vertex_num, states, symbol_set)
        }
    };
}

mod test {
    #![allow(unused)]
    use std::collections::HashSet;

    use crate::tokenizer::nfa::NFA;

    #[derive(Debug, Clone, Copy)]
    enum Token {
        Num,
        Op
    }

    #[test]
    fn test_or() {
        let nfa_1 = NFA::from_symbol('1');
        let nfa_2 = NFA::from_symbol('2');
        let mut nfa = nfa_1 | nfa_2;
        nfa.set_state(Token::Num);
        println!("{:?}", nfa);
    }

    #[test]
    fn test_and() {
        let nfa_1 = NFA::from_symbol('1');
        let nfa_2 = NFA::from_symbol('2');
        let mut nfa = nfa_1 & nfa_2;
        nfa.set_state(Token::Num);
        println!("{:?}", nfa);
    }

    #[test]
    fn test_link() {
        let mut nfa_1 = NFA::from_symbol('a');
        nfa_1.set_state('a');
        let mut nfa_2 = NFA::from_symbol('b');
        nfa_2.set_state('b');
        let mut nfa_3 = NFA::from_symbol('c');
        nfa_3.set_state('c');
        let mut nfa_4 = NFA::from_symbol('d');
        nfa_4.set_state('d');
        let mut nfa_5 = NFA::from_symbol('e');
        nfa_5.set_state('e');
        let a = link_nfa!(nfa_1, nfa_2, nfa_3, nfa_4, nfa_5);
        println!("{:?}", a);
    }

    #[test]
    fn test_e_closure() {
        //
        // 0 -> 1 - 'a' -> 2 -> 3 - 'b' -> 4 -> 5
        //              <-
        let nfa_a = NFA::from_symbol('a');
        let nfa_b = NFA::from_symbol('b');
        let mut nfa = nfa_a & nfa_b;
        nfa =  nfa.closure();
        nfa.set_state(Token::Num);
        let mut set = HashSet::new();
        set.insert(0);
        println!("e_closure(0) = {:?}", nfa.e_closure(&set));
        let mut set = HashSet::new();
        set.insert(1);
        println!("e_closure(1) = {:?}", nfa.e_closure(&set));
        let mut set = HashSet::new();
        set.insert(2);
        println!("e_closure(2) = {:?}", nfa.e_closure(&set));
        let mut set = HashSet::new();
        set.insert(3);
        println!("e_closure(3) = {:?}", nfa.e_closure(&set));
        let mut set = HashSet::new();
        set.insert(4);
        println!("e_closure(4) = {:?}", nfa.e_closure(&set));
        let mut set = HashSet::new();
        set.insert(5);
        println!("e_closure(5) = {:?}", nfa.e_closure(&set));
    }
}
