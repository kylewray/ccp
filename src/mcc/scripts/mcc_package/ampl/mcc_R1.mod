model;


# Note: These are all factors, not the full sets...
set STATES;
set ACTIONS;
set OBSERVATIONS;
set CONTROLLER_NODES;


param q0 {q1 in CONTROLLER_NODES, q2 in CONTROLLER_NODES}
  #default 0.0, >= 0.0, <= 1.0;
  default if q1 = "node1" and q2 = "node1" then 1.0 else 0.0, >= 0.0, <= 1.0;

param b0 {s1 in STATES, s2 in STATES}
  #default 0.0, >= 0.0, <= 1.0;
  default if s1 = "bl" and s2 = "tr" then 1.0 else 0.0, >= 0.0, <= 1.0;

param T {s1 in STATES, s2 in STATES, a1 in ACTIONS, a2 in ACTIONS, sp1 in STATES, sp2 in STATES} default 0.0, >= 0.0, <= 1.0;
param O {a1 in ACTIONS, a2 in ACTIONS, s1 in STATES, s2 in STATES, o1 in OBSERVATIONS, o2 in OBSERVATIONS} default 0.0, >= 0.0, <= 1.0;
param R0 {s1 in STATES, s2 in STATES, a1 in ACTIONS, a2 in ACTIONS} default 0.0, >= 0.0, <= 1.0;
param R1 {s1 in STATES, s2 in STATES, a1 in ACTIONS, a2 in ACTIONS} default 0.0, >= 0.0, <= 1.0;
param R2 {s1 in STATES, s2 in STATES, a1 in ACTIONS, a2 in ACTIONS} default 0.0, >= 0.0, <= 1.0;

param gamma default 0.95, >= 0.0, <= 1.0;


param delta default 0.0;
param V0Star default 0.0;
param psi2 {CONTROLLER_NODES, ACTIONS} default 0.0, >= 0.0, <= 1.0;
param eta2 {CONTROLLER_NODES, ACTIONS, OBSERVATIONS, CONTROLLER_NODES} default 0.0, >= 0.0, <= 1.0;


var V0 {CONTROLLER_NODES, CONTROLLER_NODES, STATES, STATES};
var V1 {CONTROLLER_NODES, CONTROLLER_NODES, STATES, STATES};
var psi1 {CONTROLLER_NODES, ACTIONS} >= 0.0, <= 1.0;
var eta1 {CONTROLLER_NODES, ACTIONS, OBSERVATIONS, CONTROLLER_NODES} >= 0.0, <= 1.0;


# Objective Function.

maximize Value:
  sum {q1 in CONTROLLER_NODES, q2 in CONTROLLER_NODES, s1 in STATES, s2 in STATES} q0[q1, q2] * b0[s1, s2] * V1[q1, q2, s1, s2];


# Bellman Constraints.

subject to Bellman_Constraint_V0 {q1 in CONTROLLER_NODES, q2 in CONTROLLER_NODES, s1 in STATES, s2 in STATES}:
  V0[q1, q2, s1, s2] = sum {a1 in ACTIONS, a2 in ACTIONS} ( psi1[q1, a1] * psi2[q2, a2] * ( R0[s1, s2, a1, a2] + gamma * sum {sp1 in STATES, sp2 in STATES} ( T[s1, s2, a1, a2, sp1, sp2] * sum {o1 in OBSERVATIONS, o2 in OBSERVATIONS} ( O[a1, a2, sp1, sp2, o1, o2] * sum {qp1 in CONTROLLER_NODES, qp2 in CONTROLLER_NODES} ( eta1[q1, a1, o1, qp1] * eta2[q2, a2, o2, qp2] * V0[qp1, qp2, sp1, sp2] ) ) ) ) );

subject to Bellman_Constraint_V1 {q1 in CONTROLLER_NODES, q2 in CONTROLLER_NODES, s1 in STATES, s2 in STATES}:
  V1[q1, q2, s1, s2] = sum {a1 in ACTIONS, a2 in ACTIONS} ( psi1[q1, a1] * psi2[q2, a2] * ( R0[s1, s2, a1, a2] + gamma * sum {sp1 in STATES, sp2 in STATES} ( T[s1, s2, a1, a2, sp1, sp2] * sum {o1 in OBSERVATIONS, o2 in OBSERVATIONS} ( O[a1, a2, sp1, sp2, o1, o2] * sum {qp1 in CONTROLLER_NODES, qp2 in CONTROLLER_NODES} ( eta1[q1, a1, o1, qp1] * eta2[q2, a2, o2, qp2] * V1[qp1, qp2, sp1, sp2] ) ) ) ) );


# Slack Constraint.

subject to Slack_Constraint:
  V0Star - sum {q1 in CONTROLLER_NODES, q2 in CONTROLLER_NODES, s1 in STATES, s2 in STATES} q0[q1, q2] * b0[s1, s2] * V0[q1, q2, s1, s2] <= delta;


# Probability Constraints - Agent 1.

subject to Probability_Constraint_Psi1_Nonnegative {q in CONTROLLER_NODES, a in ACTIONS}:
  psi1[q, a] >= 0.0;

subject to Probability_Constraint_Psi1_Normalization {q in CONTROLLER_NODES}:
  sum {a in ACTIONS} psi1[q, a] = 1.0;

subject to Probability_Constraint_Eta1_Nonnegative {q in CONTROLLER_NODES, a in ACTIONS, o in OBSERVATIONS, qp in CONTROLLER_NODES}:
  eta1[q, a, o, qp] >= 0.0;

subject to Probability_Constraint_Eta1_Normalization {q in CONTROLLER_NODES, a in ACTIONS, o in OBSERVATIONS}:
  sum {qp in CONTROLLER_NODES} eta1[q, a, o, qp] = 1.0;

