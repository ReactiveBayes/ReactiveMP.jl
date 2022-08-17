## Assessing People’s Skills

This demo demonstrates the capabilities of ReactiveMP.jl to perform inference in the models composed of Bernoulli random variables.

The demo is inspired by the example from Chapter 2 of Bishop's Model-Based Machine Learning book.
We are going to perform an exact inference to assess the skills of a student given the results of the test.

Let us assume that our imaginary test is composed of three questions, and each of these questions is associated with test results $r$, where $\{r \in \mathbb{R}, 0 < r < 1\}$

The result of the first question will solely depend on the student's attendance. For example, if the student attends the lectures, he will most certainly answer the first question.
The result of the second question will depend on a specific skill $s_2$. However, if the student has attended the lectures, he would still have a good chance of answering the second question.
We will model this relationship through disjunction or logical $OR$.
The third question is more difficult to answer, i.e., the student needs to have a particular skill $s_3$ __and__ he must have good attendance or must have a $s_3$
Hence, to model this relationship between skills and the third question, we will use conjunction or logical $AND$.

For the sake of the example, we will replace attendance with laziness. The convention is that if a person is not lazy, he attends lectures.
This way, the first question can be answered if the student is not lazy. We will use the $NOT$ function to represent this relationship.

Let us define the generative model:
$$p(l, s_2, s_3, r_1, r_2, r_3)=p(l)p(s_2)p(s_3)p(r_1|f_1(l))p(r_2|f_2(l, s_2))p(r_3|f_3(l, s_2, s_3))$$

The factors $p(l), p(s_2), p(s_3)$ represent Bernoulli prior distributions. 

$f_1(l) = NOT(l)$ where $NOT(X) \triangleq \overline{X}$, 

$f_2(l, s_2) = OR(NOT(l), s_2)$ where $OR(X, Y) \triangleq X \vee Y$, 

$f_3(l, s_2, s_3) = AND(OR(NOT(l), s_2), s_3)$ where $AND(X, Y) \triangleq X \land Y$

An attentive reader may notice that $f_2(l, s_2)$ can be rewritten as $IMPLY(l, s_2)$, i.e., $l\implies s_2$ 

Similar to the example from the Model-Based Machine Learning book, our observations are noisy. It means that the likelihood functions should map $\{0, 1\}$ to a real value $r \in (0, 1)$, denoting the result of the test. We can associate $r=0$ and $r=1.0$ with $0\%$ and $100\%$ correctness of the test, respectively.

One way of specifying the likelihood is $$p(r_i|f) = \begin{cases} r_i & \text{if }f_i = 1 \\
1-r_i & \text{if }f_i=0 \end{cases}$$
or $$p(r_i|f)=r_if_i+(1-r_i)(1-f_i)$$

It can be shown that given the observation $r_i$, the backward message from the node $p(r_i|f_i)$ will be a Bernoulli distribution with parameter $r_i$, i.e. $\overleftarrow{\mu}({f_i})\propto\mathrm{Ber}(r_i)$. 
If we observe $r_i=0.9$ it is more "likely" that the variable $f_i=1$.

Following Bishop, we will call this node function __AddNoise__

```@example skills
using Rocket, GraphPPL, ReactiveMP, Distributions, Random
```

```@example skills
# Create AddNoise node
struct AddNoise end

@node AddNoise Stochastic [out, in]
```

```@example skills
# Adding update rule for AddNoise node
@rule AddNoise(:in, Marginalisation) (q_out::PointMass,) = begin     
    return Bernoulli(mean(q_out))
end
```

```@example skills
# GraphPPL.jl export `@model` macro for model specification
# It accepts a regular Julia function and builds an FFG under the hood
@model function skill_model()

    res = datavar(Float64, 3)

    laziness ~ Bernoulli(0.5)
    skill2 ~ Bernoulli(0.5)
    skill3 ~ Bernoulli(0.5)

    test2 ~ laziness -> skill2
    test3 ~ test2 && skill3
    
    res[1] ~ AddNoise(¬laziness)
    res[2] ~ AddNoise(test2)
    res[3] ~ AddNoise(test3)

end
```

Let us assume that a student scoared $70\%$ and $95\%$ at first and second tests respectively. But got only $30\%$ on the third one. 

```@example skills
test_results = [0.7, 0.95, 0.3]

inference_result = inference(
    model = Model(skill_model),
    data  = (res = test_results, )
)
```
The results make sense. On the one hand, the student answered the first question correctly, which immediately gives us reason to believe that he is not lazy. He answered the second question pretty well, but this does not mean that the student had the skills to answer this question (attendance,i.e., lack of laziness, could help). To answer the third question, it was necessary to answer the second and have additional skills (#3). Unfortunately, the student's answer was weak, so our confidence about skill #3 was shattered.