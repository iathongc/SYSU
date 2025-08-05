(define (domain puzzle)
  (:requirements :strips :equality
                 :typing:universal-preconditions
                 :conditional-effects)
  (:types num loc) 
  (:predicates  (neighbor ?x ?y - loc)
                (at ?x - num ?y - loc)
                (zero_at ?x - loc)
      
  )

(:action slide
             :parameters (?n - num ?x ?y -loc)
             :precondition (and(zero_at ?y)(at ?n ?x)(neighbor ?x ?y))
             :effect (and(not(zero_at ?y))(zero_at ?x)(at ?n ?y)(not (at ?n ?x)))
 )
)