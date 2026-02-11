---------------------------- MODULE AuditLogIntegrity ----------------------------
(****************************************************************************)
(* TLA+ Specification for Audit Log Integrity                               *)
(*                                                                          *)
(* Verifies critical audit log properties:                                  *)
(* - Append-only (no deletions except time-based cleanup)                   *)
(* - Entries are immutable once created                                     *)
(* - Timestamps are monotonically increasing                                *)
(* - Every auditable operation generates an entry                           *)
(* - Entry IDs are unique                                                   *)
(****************************************************************************)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    Users,              \* Set of users
    Tenants,            \* Set of tenants
    MaxEntries,         \* Maximum log entries
    MaxTime,            \* Maximum simulation time
    RetentionTime       \* Entries older than this can be cleaned

VARIABLES
    auditLog,           \* Sequence of audit entries
    time,               \* Current time
    entryIds,           \* Set of all entry IDs ever created
    operations,         \* Count of operations performed
    auditedOps          \* Count of operations that were audited

(****************************************************************************)
(* Type Definitions                                                         *)
(****************************************************************************)

Actions == {"create", "delete"}

AuditEntry == [
    id: 1..5,
    timestamp: 0..MaxTime,
    user_id: Users,
    tenant_id: Tenants,
    action: Actions,
    immutable: BOOLEAN     \* Marker to verify immutability
]

(****************************************************************************)
(* Helper Functions                                                         *)
(****************************************************************************)

\* Get all entry IDs in the log
LogIds == {auditLog[i].id : i \in 1..Len(auditLog)}

\* Get entries for a specific tenant
TenantEntries(t) == {i \in 1..Len(auditLog) : auditLog[i].tenant_id = t}

\* Get entries older than retention time
OldEntries == {i \in 1..Len(auditLog) : auditLog[i].timestamp < time - RetentionTime}

\* Latest timestamp in log
LatestTimestamp ==
    IF Len(auditLog) = 0 THEN 0
    ELSE auditLog[Len(auditLog)].timestamp

(****************************************************************************)
(* Type Invariant                                                           *)
(****************************************************************************)

TypeInvariant ==
    /\ Len(auditLog) <= MaxEntries
    /\ time \in 0..MaxTime
    /\ operations \in 0..1000
    /\ auditedOps \in 0..1000

(****************************************************************************)
(* Safety Invariants                                                        *)
(****************************************************************************)

\* All entry IDs are unique
UniqueIds ==
    Cardinality(LogIds) = Len(auditLog)

\* Timestamps are monotonically increasing (within entries)
MonotonicTimestamps ==
    \A i, j \in 1..Len(auditLog):
        i < j => auditLog[i].timestamp <= auditLog[j].timestamp

\* Every entry that was created still exists (until cleanup)
\* Entries are only removed if they're past retention
NoUnexpectedDeletions ==
    \A id \in entryIds:
        \/ id \in LogIds
        \/ \E i \in 1..Len(auditLog):
            /\ auditLog[i].id = id
            /\ auditLog[i].timestamp < time - RetentionTime

\* Entries are immutable - all entries have immutable=TRUE
EntriesImmutable ==
    \A i \in 1..Len(auditLog):
        auditLog[i].immutable = TRUE

\* Operations are audited - every op has a corresponding audit entry
OperationsAudited ==
    auditedOps = operations

\* Entry IDs are always positive
ValidEntryIds ==
    \A i \in 1..Len(auditLog):
        auditLog[i].id > 0

\* Log can only grow (or shrink via cleanup of old entries only)
\* This is checked via the NoUnexpectedDeletions invariant

(****************************************************************************)
(* Initial State                                                            *)
(****************************************************************************)

Init ==
    /\ auditLog = <<>>
    /\ time = 0
    /\ entryIds = {}
    /\ operations = 0
    /\ auditedOps = 0

(****************************************************************************)
(* Actions                                                                  *)
(****************************************************************************)

\* Time advances
Tick ==
    /\ time < MaxTime
    /\ time' = time + 1
    /\ UNCHANGED <<auditLog, entryIds, operations, auditedOps>>

\* Perform an operation and audit it (the correct path)
PerformAuditedOperation(user, tenant, action, entryId) ==
    /\ Len(auditLog) < MaxEntries
    /\ entryId \notin entryIds
    /\ LET entry == [
           id |-> entryId,
           timestamp |-> time,
           user_id |-> user,
           tenant_id |-> tenant,
           action |-> action,
           immutable |-> TRUE
       ]
       IN /\ auditLog' = Append(auditLog, entry)
          /\ entryIds' = entryIds \cup {entryId}
    /\ operations' = operations + 1
    /\ auditedOps' = auditedOps + 1
    /\ UNCHANGED time

\* Clean up old entries (time-based retention)
CleanupOldEntries ==
    /\ OldEntries # {}
    /\ time > RetentionTime
    /\ LET keepIndices == {i \in 1..Len(auditLog) : auditLog[i].timestamp >= time - RetentionTime}
           newLog == [i \in 1..Cardinality(keepIndices) |->
                      auditLog[CHOOSE j \in keepIndices :
                               Cardinality({k \in keepIndices : k < j}) = i - 1]]
       IN auditLog' = newLog
    /\ UNCHANGED <<time, entryIds, operations, auditedOps>>

(****************************************************************************)
(* Invalid Actions (should never happen - for verification)                 *)
(****************************************************************************)

\* Try to modify an existing entry (should be impossible)
\* We don't include this in Next - it's here to show what's NOT allowed
ModifyEntry(i) ==
    /\ i \in 1..Len(auditLog)
    /\ auditLog' = [auditLog EXCEPT ![i].immutable = FALSE]
    /\ UNCHANGED <<time, entryIds, operations, auditedOps>>

\* Try to delete a non-old entry (should be impossible)
\* We don't include this in Next - it's here to show what's NOT allowed
DeleteRecentEntry(i) ==
    /\ i \in 1..Len(auditLog)
    /\ auditLog[i].timestamp >= time - RetentionTime
    /\ auditLog' = SubSeq(auditLog, 1, i-1) \o SubSeq(auditLog, i+1, Len(auditLog))
    /\ UNCHANGED <<time, entryIds, operations, auditedOps>>

(****************************************************************************)
(* Next State Relation                                                      *)
(****************************************************************************)

Next ==
    \/ Tick
    \/ \E u \in Users, t \in Tenants, a \in Actions, id \in 1..5:
        PerformAuditedOperation(u, t, a, id)
    \/ CleanupOldEntries

(****************************************************************************)
(* Specification                                                            *)
(****************************************************************************)

vars == <<auditLog, time, entryIds, operations, auditedOps>>

Spec == Init /\ [][Next]_vars

(****************************************************************************)
(* State Constraint (for bounded model checking)                            *)
(****************************************************************************)

StateConstraint ==
    /\ Len(auditLog) <= 3
    /\ time <= 4

=============================================================================
