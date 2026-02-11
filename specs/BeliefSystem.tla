---------------------------- MODULE BeliefSystem ----------------------------
(***************************************************************************)
(* TLA+ Specification for the Belief System                                *)
(*                                                                         *)
(* Models belief management including:                                     *)
(* - Confidence levels and updates                                         *)
(* - Contradiction detection                                               *)
(* - Evidence tracking                                                     *)
(* - Status transitions                                                    *)
(***************************************************************************)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    Entities,           \* Set of possible entities
    Predicates,         \* Set of possible predicates
    MaxBeliefs,         \* Maximum number of beliefs
    ConfidenceCap       \* Maximum confidence for derived beliefs (e.g., 80)

VARIABLES
    beliefs,            \* belief_id -> belief record
    contradictions,     \* Set of contradiction pairs
    nextId              \* Next belief ID

(***************************************************************************)
(* Type Definitions                                                        *)
(***************************************************************************)

Triplet == [
    subject: Entities,
    predicate: Predicates,
    object: Entities
]

Status == {"active", "contested", "rejected"}

ConfidenceValues == {0, 50, 80, 100}  \* Simplified confidence levels for model checking

Belief == [
    id: 1..MaxBeliefs,
    triplet: Triplet,
    confidence: 0..100,
    status: Status,
    evidence_for: SUBSET (1..MaxBeliefs),
    evidence_against: SUBSET (1..MaxBeliefs),
    derived: BOOLEAN          \* True if created by reflection
]

(***************************************************************************)
(* Helper Functions                                                        *)
(***************************************************************************)

\* Check if two triplets contradict (same subject+predicate, different object)
Contradicts(t1, t2) ==
    /\ t1.subject = t2.subject
    /\ t1.predicate = t2.predicate
    /\ t1.object # t2.object

\* Find all beliefs that contradict a given belief
FindContradictions(b) ==
    {b2 \in DOMAIN beliefs:
        /\ b2 # b
        /\ beliefs[b2].status # "rejected"
        /\ Contradicts(beliefs[b].triplet, beliefs[b2].triplet)}

\* Calculate effective confidence cap for a belief
EffectiveConfidenceCap(b) ==
    IF beliefs[b].derived THEN ConfidenceCap ELSE 100

(***************************************************************************)
(* Type Invariant                                                          *)
(***************************************************************************)

TypeInvariant ==
    /\ \A i \in DOMAIN beliefs: beliefs[i] \in Belief
    /\ Len(beliefs) <= MaxBeliefs
    /\ contradictions \subseteq (1..MaxBeliefs) \X (1..MaxBeliefs)
    /\ nextId \in 1..(MaxBeliefs + 1)

(***************************************************************************)
(* Safety Invariants                                                       *)
(***************************************************************************)

\* Confidence is within valid range
ConfidenceValid ==
    \A b \in DOMAIN beliefs:
        beliefs[b].confidence >= 0 /\ beliefs[b].confidence <= 100

\* Derived beliefs respect confidence cap
DerivedConfidenceCapped ==
    \A b \in DOMAIN beliefs:
        beliefs[b].derived => beliefs[b].confidence <= ConfidenceCap

\* Contradictions are symmetric
ContradictionsSymmetric ==
    \A <<b1, b2>> \in contradictions:
        <<b2, b1>> \in contradictions

\* Contradictions imply contested status
ContradictionsContested ==
    \A <<b1, b2>> \in contradictions:
        /\ beliefs[b1].status \in {"contested", "rejected"}
        /\ beliefs[b2].status \in {"contested", "rejected"}

\* Evidence sets don't contain self-references
NoSelfEvidence ==
    \A b \in DOMAIN beliefs:
        /\ b \notin beliefs[b].evidence_for
        /\ b \notin beliefs[b].evidence_against

(***************************************************************************)
(* State Constraint - limit exploration depth                              *)
(***************************************************************************)

StateConstraint == Len(beliefs) <= MaxBeliefs

(***************************************************************************)
(* Initial State                                                           *)
(***************************************************************************)

Init ==
    /\ beliefs = <<>>
    /\ contradictions = {}
    /\ nextId = 1

(***************************************************************************)
(* Actions                                                                 *)
(***************************************************************************)

\* Store a new belief
StoreBelief(triplet, confidence, derived) ==
    /\ Cardinality(DOMAIN beliefs) < MaxBeliefs
    /\ LET cappedConf == IF derived
                         THEN IF confidence > ConfidenceCap
                              THEN ConfidenceCap
                              ELSE confidence
                         ELSE confidence
           newBelief == [
               id |-> nextId,
               triplet |-> triplet,
               confidence |-> cappedConf,
               status |-> "active",
               evidence_for |-> {},
               evidence_against |-> {},
               derived |-> derived
           ]
       IN /\ beliefs' = beliefs @@ (nextId :> newBelief)
          /\ nextId' = nextId + 1
    /\ contradictions' = contradictions

\* Detect and record contradictions after storing
DetectContradictions ==
    /\ \E b1, b2 \in DOMAIN beliefs:
        /\ b1 < b2
        /\ <<b1, b2>> \notin contradictions
        /\ beliefs[b1].status # "rejected"
        /\ beliefs[b2].status # "rejected"
        /\ Contradicts(beliefs[b1].triplet, beliefs[b2].triplet)
        /\ contradictions' = contradictions \cup {<<b1, b2>>, <<b2, b1>>}
        /\ beliefs' = [beliefs EXCEPT
            ![b1].status = "contested",
            ![b2].status = "contested"]
    /\ UNCHANGED nextId

\* Add supporting evidence
AddSupportingEvidence(belief_id, evidence_id) ==
    /\ belief_id \in DOMAIN beliefs
    /\ evidence_id \in DOMAIN beliefs
    /\ belief_id # evidence_id
    /\ evidence_id \notin beliefs[belief_id].evidence_for
    /\ beliefs' = [beliefs EXCEPT
        ![belief_id].evidence_for = @ \cup {evidence_id}]
    /\ UNCHANGED <<contradictions, nextId>>

\* Add contradicting evidence
AddContradictingEvidence(belief_id, evidence_id) ==
    /\ belief_id \in DOMAIN beliefs
    /\ evidence_id \in DOMAIN beliefs
    /\ belief_id # evidence_id
    /\ evidence_id \notin beliefs[belief_id].evidence_against
    /\ beliefs' = [beliefs EXCEPT
        ![belief_id].evidence_against = @ \cup {evidence_id}]
    /\ UNCHANGED <<contradictions, nextId>>

\* Increase confidence based on evidence
IncreaseConfidence(id, amount) ==
    /\ id \in DOMAIN beliefs
    /\ beliefs[id].status = "active"
    /\ LET cap == EffectiveConfidenceCap(id)
           newConf == IF beliefs[id].confidence + amount > cap
                      THEN cap
                      ELSE beliefs[id].confidence + amount
       IN beliefs' = [beliefs EXCEPT ![id].confidence = newConf]
    /\ UNCHANGED <<contradictions, nextId>>

\* Decrease confidence
DecreaseConfidence(id, amount) ==
    /\ id \in DOMAIN beliefs
    /\ LET newConf == IF beliefs[id].confidence < amount
                      THEN 0
                      ELSE beliefs[id].confidence - amount
       IN beliefs' = [beliefs EXCEPT ![id].confidence = newConf]
    /\ UNCHANGED <<contradictions, nextId>>

\* Reject a belief (e.g., due to overwhelming counter-evidence)
RejectBelief(id) ==
    /\ id \in DOMAIN beliefs
    /\ beliefs[id].status # "rejected"
    /\ beliefs' = [beliefs EXCEPT ![id].status = "rejected"]
    \* Remove from contradictions since rejected beliefs don't matter
    /\ contradictions' = {<<b1, b2>> \in contradictions: b1 # id /\ b2 # id}
    /\ UNCHANGED nextId

\* Resolve contradiction by rejecting lower confidence belief
ResolveContradiction(b1, b2) ==
    /\ <<b1, b2>> \in contradictions
    /\ beliefs[b1].confidence # beliefs[b2].confidence
    /\ LET loser == IF beliefs[b1].confidence < beliefs[b2].confidence
                    THEN b1 ELSE b2
           winner == IF loser = b1 THEN b2 ELSE b1
       IN /\ beliefs' = [beliefs EXCEPT
              ![loser].status = "rejected",
              ![winner].status = "active"]
          /\ contradictions' = {<<x, y>> \in contradictions:
              x # loser /\ y # loser}
    /\ UNCHANGED nextId

(***************************************************************************)
(* Next State Relation                                                     *)
(***************************************************************************)

Next ==
    \/ \E s, o \in Entities, p \in Predicates, c \in ConfidenceValues, d \in BOOLEAN:
        StoreBelief([subject |-> s, predicate |-> p, object |-> o], c, d)
    \/ DetectContradictions
    \/ \E b, e \in DOMAIN beliefs:
        AddSupportingEvidence(b, e)
    \/ \E b, e \in DOMAIN beliefs:
        AddContradictingEvidence(b, e)
    \/ \E b \in DOMAIN beliefs:
        RejectBelief(b)
    \/ \E b1, b2 \in DOMAIN beliefs:
        ResolveContradiction(b1, b2)

(***************************************************************************)
(* Specification                                                           *)
(***************************************************************************)

Spec == Init /\ [][Next]_<<beliefs, contradictions, nextId>>

(***************************************************************************)
(* Temporal Properties                                                     *)
(***************************************************************************)

\* Contradictions are eventually resolved
EventuallyResolved ==
    \A <<b1, b2>> \in contradictions:
        <<b1, b2>> \in contradictions ~>
            (beliefs[b1].status = "rejected" \/ beliefs[b2].status = "rejected")

\* Low confidence beliefs eventually get rejected or strengthened
EventuallyDecided ==
    \A b \in DOMAIN beliefs:
        beliefs[b].confidence < 30 ~>
            (beliefs[b].status = "rejected" \/ beliefs[b].confidence >= 30)

=============================================================================
