------------------------- MODULE TenantIsolationSimple -------------------------
(*
 * Simplified TLA+ Spec for Silicon Memory Tenant Isolation
 * Uses concrete initial state to reduce state space for model checking.
 *)

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS
    Users,
    Tenants,
    Documents,
    u1, u2,    \* User constants
    t1, t2,    \* Tenant constants
    d1         \* Document constant

VARIABLES
    userTenant,
    userRoles,
    docOwner,
    docTenant,
    docPrivacy,
    docSharedWith,
    accessLog

vars == <<userTenant, userRoles, docOwner, docTenant, docPrivacy, docSharedWith, accessLog>>

(* Helper for Range of a sequence *)
Range(s) == {s[i] : i \in DOMAIN s}

-----------------------------------------------------------------------------
(* Helper Predicates *)

IsAdmin(user) == "admin" \in userRoles[user]
IsMember(user) == "member" \in userRoles[user] \/ "viewer" \in userRoles[user]
SameTenant(user, doc) == userTenant[user] = docTenant[doc]
IsOwner(user, doc) == docOwner[doc] = user
IsExplicitlyShared(user, doc) == user \in docSharedWith[doc]

-----------------------------------------------------------------------------
(* Access Control Logic *)

CanAccess(user, doc) ==
    \/ IsOwner(user, doc)
    \/ docPrivacy[doc] = "public"
    \/ /\ SameTenant(user, doc)
       /\ \/ IsAdmin(user)
          \/ docPrivacy[doc] = "workspace" /\ IsMember(user)
          \/ IsExplicitlyShared(user, doc)

-----------------------------------------------------------------------------
(* State Constraint - limit exploration depth *)
StateConstraint == Len(accessLog) < 4

-----------------------------------------------------------------------------
(* Actions *)

AccessAttempt(user, doc) ==
    \* Include privacy level at access time to properly check invariants
    /\ accessLog' = Append(accessLog, <<user, doc, CanAccess(user, doc), docPrivacy[doc]>>)
    /\ UNCHANGED <<userTenant, userRoles, docOwner, docTenant, docPrivacy, docSharedWith>>

ChangePrivacy(doc, newPrivacy) ==
    /\ docPrivacy' = [docPrivacy EXCEPT ![doc] = newPrivacy]
    /\ UNCHANGED <<userTenant, userRoles, docOwner, docTenant, docSharedWith, accessLog>>

ShareDoc(doc, user) ==
    /\ docSharedWith' = [docSharedWith EXCEPT ![doc] = @ \cup {user}]
    /\ UNCHANGED <<userTenant, userRoles, docOwner, docTenant, docPrivacy, accessLog>>

-----------------------------------------------------------------------------
(* Safety Invariants *)

TenantIsolationInvariant ==
    \A entry \in Range(accessLog):
        LET user == entry[1]
            doc == entry[2]
            granted == entry[3]
            privacyAtAccess == entry[4]
        IN
            granted /\ ~SameTenant(user, doc) => privacyAtAccess = "public"

PrivateAccessInvariant ==
    \A entry \in Range(accessLog):
        LET user == entry[1]
            doc == entry[2]
            granted == entry[3]
            privacyAtAccess == entry[4]
        IN
            granted /\ privacyAtAccess = "private" =>
                \/ IsOwner(user, doc)
                \/ (SameTenant(user, doc) /\ IsAdmin(user))
                \/ IsExplicitlyShared(user, doc)

SafetyInvariant ==
    /\ TenantIsolationInvariant
    /\ PrivateAccessInvariant

-----------------------------------------------------------------------------
(* Concrete Initial State - reduces state space *)

Init ==
    \* Users can be in same or different tenants
    /\ userTenant \in [Users -> Tenants]
    \* Users can be members or admins
    /\ userRoles \in [Users -> {{"member"}, {"admin"}}]
    \* Document owned by first user
    /\ docOwner = [d \in Documents |-> u1]
    /\ docTenant = [d \in Documents |-> userTenant[docOwner[d]]]
    \* Document can have any privacy level
    /\ docPrivacy \in [Documents -> {"private", "workspace", "public"}]
    /\ docSharedWith = [d \in Documents |-> {}]
    /\ accessLog = <<>>

-----------------------------------------------------------------------------
(* Next State *)

Next ==
    \/ \E user \in Users, doc \in Documents: AccessAttempt(user, doc)
    \/ \E doc \in Documents, privacy \in {"private", "workspace", "public"}: ChangePrivacy(doc, privacy)
    \/ \E doc \in Documents, user \in Users: ShareDoc(doc, user)

-----------------------------------------------------------------------------
(* Specification *)

Spec == Init /\ [][Next]_vars

=============================================================================
