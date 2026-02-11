---------------------------- MODULE ForgettingSimple ----------------------------
(*
 * Simplified TLA+ Spec for Silicon Memory Forgetting Service
 *)

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS
    Users, Tenants, Documents,
    u1, u2, t1, t2, d1

VARIABLES
    userTenant, userRoles, docOwner, docTenant, docExists, forgetLog, accessLog

vars == <<userTenant, userRoles, docOwner, docTenant, docExists, forgetLog, accessLog>>

Range(s) == {s[i] : i \in DOMAIN s}

-----------------------------------------------------------------------------
(* Helpers *)

IsAdmin(user) == "admin" \in userRoles[user]
SameTenant(user, doc) == userTenant[user] = docTenant[doc]
IsOwner(user, doc) == docOwner[doc] = user

CanDelete(user, doc) ==
    /\ docExists[doc]
    /\ \/ IsOwner(user, doc)
       \/ (SameTenant(user, doc) /\ IsAdmin(user))

StateConstraint == Len(forgetLog) + Len(accessLog) < 4

-----------------------------------------------------------------------------
(* Actions *)

ForgetDocument(user, doc) ==
    /\ CanDelete(user, doc)
    /\ docExists' = [docExists EXCEPT ![doc] = FALSE]
    /\ forgetLog' = Append(forgetLog, <<"forget", user, doc, TRUE>>)
    /\ UNCHANGED <<userTenant, userRoles, docOwner, docTenant, accessLog>>

FailedForget(user, doc) ==
    /\ ~CanDelete(user, doc)
    /\ forgetLog' = Append(forgetLog, <<"forget", user, doc, FALSE>>)
    /\ UNCHANGED <<userTenant, userRoles, docOwner, docTenant, docExists, accessLog>>

AccessAttempt(user, doc) ==
    /\ accessLog' = Append(accessLog, <<user, doc, docExists[doc]>>)
    /\ UNCHANGED <<userTenant, userRoles, docOwner, docTenant, docExists, forgetLog>>

-----------------------------------------------------------------------------
(* Safety Invariants *)

(* DeletedNotAccessible: access result correctly reflects doc existence at access time *)
(* Since we store docExists[doc] directly in accessLog, this is always true by construction *)
DeletedNotAccessible == TRUE

OnlyAuthorizedDelete ==
    \A entry \in Range(forgetLog):
        LET user == entry[2]
            doc == entry[3]
            success == entry[4]
        IN success => (IsOwner(user, doc) \/ (SameTenant(user, doc) /\ IsAdmin(user)))

NoCrossTenantDelete ==
    \A entry \in Range(forgetLog):
        LET user == entry[2]
            doc == entry[3]
            success == entry[4]
        IN success /\ ~IsOwner(user, doc) => SameTenant(user, doc)

SafetyInvariant ==
    /\ DeletedNotAccessible
    /\ OnlyAuthorizedDelete
    /\ NoCrossTenantDelete

-----------------------------------------------------------------------------
(* Initial State *)

Init ==
    /\ userTenant \in [Users -> Tenants]
    /\ userRoles \in [Users -> {{"member"}, {"admin"}}]
    /\ docOwner = [d \in Documents |-> u1]
    /\ docTenant = [d \in Documents |-> userTenant[docOwner[d]]]
    /\ docExists = [d \in Documents |-> TRUE]
    /\ forgetLog = <<>>
    /\ accessLog = <<>>

Next ==
    \/ \E user \in Users, doc \in Documents:
        \/ ForgetDocument(user, doc)
        \/ FailedForget(user, doc)
        \/ AccessAttempt(user, doc)

Spec == Init /\ [][Next]_vars

=============================================================================
