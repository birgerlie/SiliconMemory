-------------------------- MODULE AccessControlSimple --------------------------
(*
 * Simplified TLA+ Spec for Silicon Memory ABAC Access Control
 *)

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS
    Users, Roles, Resources, Permissions, Policies,
    u1, u2, admin, member, r1, read, write, p1, p2

VARIABLES
    userRoles, resourceOwner, policyPriority, policyEffect, policyEnabled,
    accessRequests, accessDecisions

vars == <<userRoles, resourceOwner, policyPriority, policyEffect, policyEnabled,
          accessRequests, accessDecisions>>

Range(s) == {s[i] : i \in DOMAIN s}

StateConstraint == Len(accessRequests) < 3

-----------------------------------------------------------------------------
(* Policy Evaluation *)

HighestPriorityPolicy(user, resource, permission) ==
    LET enabled == {p \in Policies : policyEnabled[p]}
    IN IF enabled = {}
       THEN "none"
       ELSE CHOOSE p \in enabled:
            \A q \in enabled: policyPriority[p] >= policyPriority[q]

EvaluateAccess(user, resource, permission) ==
    LET hp == HighestPriorityPolicy(user, resource, permission)
    IN IF hp = "none"
       THEN "deny"  \* Default deny
       ELSE policyEffect[hp]

OwnerAccess(user, resource) == resourceOwner[resource] = user

FinalAccess(user, resource, permission) ==
    \/ OwnerAccess(user, resource)
    \/ EvaluateAccess(user, resource, permission) = "allow"

-----------------------------------------------------------------------------
(* Actions *)

AccessRequest(user, resource, permission) ==
    /\ accessRequests' = Append(accessRequests, <<user, resource, permission>>)
    /\ accessDecisions' = Append(accessDecisions,
                                  <<user, resource, permission,
                                    FinalAccess(user, resource, permission)>>)
    /\ UNCHANGED <<userRoles, resourceOwner, policyPriority, policyEffect, policyEnabled>>

TogglePolicy(policy) ==
    /\ policyEnabled' = [policyEnabled EXCEPT ![policy] = ~policyEnabled[policy]]
    /\ UNCHANGED <<userRoles, resourceOwner, policyPriority, policyEffect,
                   accessRequests, accessDecisions>>

-----------------------------------------------------------------------------
(* Safety Invariants *)

(* Access control is deterministic *)
Deterministic ==
    \A i, j \in DOMAIN accessDecisions:
        /\ accessDecisions[i][1] = accessDecisions[j][1]  \* Same user
        /\ accessDecisions[i][2] = accessDecisions[j][2]  \* Same resource
        /\ accessDecisions[i][3] = accessDecisions[j][3]  \* Same permission
        => accessDecisions[i][4] = accessDecisions[j][4]  \* Same decision

(* Owners always have access *)
OwnersHaveAccess ==
    \A entry \in Range(accessDecisions):
        LET user == entry[1]
            resource == entry[2]
            granted == entry[4]
        IN OwnerAccess(user, resource) => granted

SafetyInvariant ==
    /\ OwnersHaveAccess

-----------------------------------------------------------------------------
(* Initial State *)

Init ==
    /\ userRoles \in [Users -> SUBSET Roles]
    /\ resourceOwner = [r \in Resources |-> u1]
    /\ policyPriority = [p \in Policies |-> IF p = p1 THEN 10 ELSE 5]
    /\ policyEffect \in [Policies -> {"allow", "deny"}]
    /\ policyEnabled = [p \in Policies |-> TRUE]
    /\ accessRequests = <<>>
    /\ accessDecisions = <<>>

Next ==
    \/ \E user \in Users, resource \in Resources, permission \in Permissions:
        AccessRequest(user, resource, permission)
    \/ \E policy \in Policies:
        TogglePolicy(policy)

Spec == Init /\ [][Next]_vars

=============================================================================
