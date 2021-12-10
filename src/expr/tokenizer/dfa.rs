use std::{collections::{ HashMap, HashSet, VecDeque }, hash::Hash};

use super::{nfa::NFA, token::{Token, TokenType}};

#[derive(PartialEq, Eq)]
struct VertexSet {
    inner: HashSet<u32>,
}

impl Hash for VertexSet {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let mut vec = Vec::with_capacity(self.inner.len());
        for v in &self.inner {
            vec.push(*v);
        }
        vec.sort();
        vec.iter().for_each(|&v| v.hash(state));
    }
}

impl From<HashSet<u32>> for VertexSet {
    fn from(inner: HashSet<u32>) -> Self {
        Self { inner }
    }
}

impl Into<HashSet<u32>> for VertexSet {
    fn into(self) -> HashSet<u32> {
        self.inner
    }
}

#[derive(Debug, Clone, Copy)]
struct DFAEdge(u32, u32, char);

#[derive(Debug)]
pub struct DFA<AcceptEnum> {
    edges: Vec<DFAEdge>,
    states: HashMap<u32, AcceptEnum>,
}

impl<Ac: Copy> DFA<Ac> {
    #[allow(unused)]
    pub fn match_one(&self, words: &str) -> Option<Ac> {
        let mut current_vertex = 0;
        let mut is_move;
        for symbol in words.chars() {
            is_move = false;
            for DFAEdge(start, end, s) in &self.edges {
                if *start == current_vertex && symbol == *s {
                    current_vertex = *end;
                    is_move = true;
                    break;
                }
            }
            if !is_move { return None };
        }
        return self.states.get(&current_vertex).map(|s| *s);
    }
    pub fn subset_construct(nfa: &NFA<Ac>) -> Self {
        let mut dfa_edges = Vec::new();
        let mut dfa_states = HashMap::new();

        let init = nfa.e_closure_with_vertex(0);
        let mut nfa_vertex_sets = vec![VertexSet::from(init.clone())];
        let mut table = HashMap::new();
        table.insert(VertexSet::from(init), 0_u32);

        let mut unmarked_pos = 0;
        let mut dfa_vertex_num = 1;

        while unmarked_pos < nfa_vertex_sets.len() {
            for symbol in nfa.symbol_set() {
                let move_set = nfa.move_set(&nfa_vertex_sets[unmarked_pos].inner, *symbol);
                let nfa_vertex_set = nfa.e_closure(&move_set);
                if nfa_vertex_set.is_empty() { continue; }
                let set_warp = VertexSet::from(nfa_vertex_set.clone());
                match table.get(&set_warp) {
                    Some(&dfa_vertex_num) => {
                        let start = *table.get(&nfa_vertex_sets[unmarked_pos]).unwrap();
                        let end = dfa_vertex_num;
                        dfa_edges.push(DFAEdge(start, end, *symbol));
                    },
                    None => {
                        match nfa.get_state(&set_warp.inner) {
                            Some(state) => { dfa_states.insert(dfa_vertex_num, state); },
                            None => { /* Do nothing */ },
                        }

                        table.insert(set_warp, dfa_vertex_num);
                        nfa_vertex_sets.push(VertexSet::from(nfa_vertex_set));
                        let start = *table.get(&nfa_vertex_sets[unmarked_pos]).unwrap();
                        let end = dfa_vertex_num;
                        
                        dfa_edges.push(DFAEdge(start, end, *symbol));
                        dfa_vertex_num += 1;
                    },
                }
            }
            unmarked_pos += 1;
        }
        DFA { edges: dfa_edges, states: dfa_states }
    }
}

impl<Ac: TokenType> DFA<Ac>  {
    pub fn tokenize(&self, origin: &str) -> VecDeque<Token<Ac>> {
        let mut result = VecDeque::new();
        let mut current_vertex = 0;
        // let mut prev_vertex = 0;
        let mut is_move;
        let mut val = String::new();
        let mut current_pos = 0;
        let chars = origin.chars().collect::<Vec<_>>();
        let chars_len = chars.len();
        while current_pos < chars_len {
            let symbol = unsafe { *chars.get_unchecked(current_pos) };
            is_move = false;
            for DFAEdge(start, end, s) in &self.edges {
                if *start == current_vertex && symbol == *s {
                    val.push(symbol);
                    // prev_vertex = current_vertex;
                    current_vertex = *end;
                    is_move = true;
                    current_pos += 1;
                    break;
                }
            }
            if !is_move {
                match self.states.get(&current_vertex) {
                    Some(ac) => {
                        if !ac.is_nonterminal() {
                            result.push_back(Token::new(val.clone(), *ac));
                        }
                        current_vertex = 0;
                    },
                    None => panic!("unexpected symbol `{}` in `{}`. ", symbol, val),
                }
                val.clear();
            }
        }
        match self.states.get(&current_vertex) {
            Some(ac) => {
                if !ac.is_nonterminal() {
                    result.push_back(Token::new(val.clone(), *ac));
                }
            },
            None => panic!("unexpected symbol `{}` in `{}`. ", unsafe { *chars.get_unchecked(current_pos - 1) }, val),
        }
        result
    }
}

mod test {
    #![allow(unused)]
    use crate::link_nfa;
    use crate::expr::tokenizer::nfa::NFA;
    use crate::expr::tokenizer::dfa::DFA;

    #[derive(Clone, Copy, Debug)]
    enum Token {
        Num,
        Alp,
    }

    #[test]
    fn test_match() {
        let digit = NFA::from_symbol_range('0'..='9');
        let dot = NFA::from_symbol('.');
        let mut float = (digit.clone() & dot) & digit.clone().closure();
        let no_zero = NFA::from_symbol_range('1'..='9');
        let int = no_zero & digit.clone().closure();
        let mut num = float | int;
        num.set_state(Token::Num);

        let alp_upcase = NFA::from_symbol_range('A'..='Z');
        let alp_downcase = NFA::from_symbol_range('a'..='z');
        let mut alp_nfa = (alp_upcase | alp_downcase).closure();
        alp_nfa.set_state(Token::Alp);

        let tokenizer_nfa = link_nfa!(num, alp_nfa);

        let tokenizer = DFA::subset_construct(&tokenizer_nfa);
        match tokenizer.match_one("edqdsafwSWfwf") {
            Some(token) => { println!("{:?}", token) },
            None => println!("Reject"),
        }
    }
}
