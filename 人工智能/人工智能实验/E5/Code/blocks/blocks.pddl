(define (domain blocks)
  (:requirements :strips :typing :equality
                 :universal-preconditions
                 :conditional-effects)
  (:types physob)
  (:predicates   
      (ontable ?x - physob)
      (clear ?x - physob) 
      (on ?x ?y - physob))
  
  (:action move
    :parameters (?x ?y - physob)
    :precondition (and (clear ?x) (clear ?y) (not (= ?x ?y)))
    :effect (and 
              (forall (?z - physob)
                (when (on ?x ?z)
                  (and (clear ?z) (not (on ?x ?z)))))
              (on ?x ?y)
              (not (clear ?y))
              (not (ontable ?x))))
             
  (:action moveToTable
    :parameters (?x - physob)
    :precondition (and (not (ontable ?x)) (clear ?x))
    :effect (and 
              (forall (?y - physob)
                (when (on ?x ?y)
                  (and (clear ?y) (not (on ?x ?y)))))
              (ontable ?x)))
)