# Vehicle scheduling under stochastic trip times: An approximate dynamic programming approach

* author: Fang Hea, Jie Yang, Meng Li
* Keywords:
  * Vehicle scheduling
  * Stochastic trip
  * times Delay propagation
  * Approximate dynamic programming

## ABSTRACT

Due to unexpected demand surge and supply disruptions, road traffic conditions could exhibit substantial uncertainty, which often makes bus travelers encounter start delays of service trips and substantially degrades the performance of an urban transit system.
Meanwhile, rapid advances of information and communication technologies have presented tremendous opportunities for intelligently scheduling a bus fleet.
With the full consideration of delay propagation effects, this paper is devoted to formulating the stochastic dynamic vehicle scheduling problem, which dynamically schedules an urban bus fleet to tackle the trip time stochasticity, reduce the delay and minimize the total costs of a transit system.
To address the challenge of “curse of dimensionality”, we adopt an approximate dynamic programming approach (ADP) where the value function is approximated through a three-layer feed-forward neural network so that we are capable of stepping forward to make decisions and solving the Bellman’s equation through sequentially solving multiple mixed integer linear programs.
Numerical examples based on the realistic operations dataset of bus lines in Beijing have demonstrated that the proposed neuralnetwork-based ADP approach not only exhibits a good learning behavior but also significantly outperforms both myopic and static polices, especially when trip time stochasticity is high.

## 1. Introduction

Vehicle scheduling plays a key role in public transit operational planning, which consists of line planning, timetabling, vehicle scheduling, and crew scheduling.
In a vehicle scheduling problem (VSP), given the detailed information of the timetable of trips, buses are scheduled to finish the tasks (trips on the timetable) with the consideration of practical requirements such as multiple depots and vehicle types, so that each task is completed by a unique bus.
In a large-scale transit system, a deadhead trip can be inserted into two adjacent trips to reduce the number of buses used.
Recently, because of the growing competition in public transport markets, guaranteeing an adequate service level has become crucial for public transport companies.
Many surveys have reported that punctuality is important to bus passengers and considered as one of the key reasons for people’s dissatisfaction with bus services (e.g., Passenger Focus, 2014; Department for Transport of UK, 2011).

Traditionally, public transport companies complete vehicle scheduling several weeks before operations, with the objective of minimizing the planned total cost, including the fixed cost of vehicles and the variable cost for idle and travel times.
However, on the day of operations, road traffic conditions could exhibit substantial uncertainty, due to unexpected demand surge and supply disruptions, which may appear when road construction and traffic accidents occur (FHWA, 2006).
Under stochastic trip times, a bus trip could be directly delayed because of traffic congestion.
Moreover, it is also possible that a late arrival of a delayed trip causes a delayed start of its following service trip.
In other words, delays can propagate along the adjacent trips fulfilled by the same bus.
Consequently, this significant variability of road traffic conditions often makes travelers encounter start delays of service trips, which substantially degrades the performance of an urban transit system.
To prevent delays in a transit system, buffer time can be introduced between different trips, which, however, will inevitably increase the bus’s idle time and lead to the increase of the system’s costs.

Nowadays, bus vehicles in many cities have been equipped with global positioning systems, which enable transit agencies to monitor real-time bus operations.
Utilizing the information from this automatic vehicle location system (AVL), many studies have been conducted to enhance the control strategies of bus operations (e.g., Bie et al., 2015; Yu et al., 2016; Berrebi et al., 2017; Du et al., 2017).
We envision that the rapid advances of information and communication technologies have also presented tremendous opportunities for intelligently scheduling a bus fleet.
Specifically, on one hand, the data gathered through the AVL can substantially contribute to learning how different vehicle schedules affect the operations of a bus fleet under stochastic trip times.
On the other hand, within an operations day, we gradually observe the actual arrival times of trips and can thus dynamically schedule the vehicles to fulfill subsequent trips.
In practice, this dynamic vehicle scheduling demands an efficient and convenient communication between transit agencies and bus drivers, which is becoming technically feasible thanks to the advances of communication technology.

Stochastic dynamic programming (SDP) is considered to be particularly applicable to the problem of sequential decision making under uncertain conditions.
With properly chosen state and decision variables, we are capable of formulating the stochastic dynamic VSP for dynamically scheduling a bus fleet.
More specifically, in the proposed stochastic dynamic vehicle scheduling framework, each time vehicles are rescheduled, we take into account not only the bus fleet’s operations within the current period but also the rescheduling’s impact on the fleet’s operations in the future, which is captured through a cost-to-go function.
However, because of trip time stochasticity, the proposed SDP’s state space’s augmentation with uncertain travel times easily leads to “curse of dimensionality”, which is widely cited as the Achilles heel of dynamic programming (e.g., Powell, 2007).
To tackle this challenge, we adopt an approximate dynamic programming (ADP) approach where the cost-to-go function is approximated so that we are capable of stepping forward to make decisions and solving the Bellman’s equation through sequentially solving multiple mixed integer linear programs.

The contributions of our paper include:
(i) we investigate the multi-depot VSP under stochastic travel times with the full consideration of delay propagation phenomenon;
(ii) this study is among the first groups to propose an ADP approach to tackle the trip time stochasticity, mitigate delays and minimize the total cost of a transit system, through dynamically scheduling vehicles;
(iii) we do not pre-assume any probability distribution or scenario of trip times and delay propagation, and the impact of scheduling strategies on bus fleet operations under stochastic trip times is directly learned from the multi-day operational dataset;
(iv) we employ the realistic operations dataset of bus lines in Beijing to test our proposed framework, and it has been shown that our neural-networkbased ADP approach not only exhibits a good learning behavior but also significantly outperforms both myopic and static polices.

For the remainder, Section 2 reviews the literature on VSP and ADP.
Section 3 formulates the stochastic dynamic VSP.
In Section 4, we propose the ADP framework.
Section 5 employs the realistic dataset to demonstrate the effectiveness of the proposed ADP framework and derive important insights.
Finally, Section 6 concludes the paper.

## 2. Literature review

### 2.1. Vehicle scheduling problems in public transport

A public transit planning process starts with collecting and forecasting passengers’ demands.
According to the demand matrices, local authorities then design the infrastructure of public transport networks and plan lines and their frequencies (timetables).
Each trip on a timetable has its own departure and arrival times as well as start and end stations.
Next, a public transport company schedules its vehicle fleet to cover these trips and assures that each scheduled trip is served by a unique vehicle, which is widely referred to as VSP.
Finally, crew scheduling needs to be performed.
It is desirable to plan all the activities above simultaneously for the purpose of maximizing the system’s productivity and efficiency.
However, because this planning process is extremely complex, especially for medium and large fleet sizes, it requires separate treatment for each activity, with the outcome of one fed as an input to the next (e.g., Ceder, 2007).

Vehicle scheduling has become an important research field for about 40–50 years (Bunte and Kliewer, 2009), which can be divided into two categories based on the number of depots.
The single-depot vehicle scheduling problem (SDVSP) indicates that only one depot is considered, whereas the multiple-depot vehicle scheduling problem (MDVSP) considers that vehicles are stationed in multiple depots.
In MDVSP, vehicles need to return to their start depots after a day’s operation, and some trips have to be assigned to vehicles from a certain set of depots.
The SDVSP is solvable in a polynomial time, and a large number of solution methods have been proposed (e.g., Saha, 1970; Gertsbach and Gurevich, 1977; Ceder, 2016).
However, the MDVSP is proved to be NP-hard (Bertossi et al., 1987).
Various strategies have been proposed to reduce the number of model variables, such as adopting the time-space diagram and column generation approach (e.g., Ribeiro and Soumis, 1994; Bodin et al., 1983).
Both exact algorithms (e.g., Kliewer et al., 2006; Oukil et al., 2007) and heuristics (e.g., Ball et al., 1983) have been developed to solve MDVSP.
We refer interested readers to Desaulniers and Hickman (2007) and Bunte and Kliewer (2009) for recent reviews.

As previously mentioned, travel time’s uncertainty will inevitably cause the delay of service trips.
In order to assure a satisfying service level for a bus system, transit system operators should consider delay cost.
Huisman et al. (2004) adopted a quadratic function to penalize delays when scheduling, and then introduced a dynamic scheduling approach by solving a sequence of optimization problems and taking into account different scenarios for future travel times.
Naumann et al. (2011) considered typical travel time scenarios during optimization and minimized the expected sum of planned travel costs and costs caused by delays.
Yan and Tang (2008) developed a framework that combined the planning and real-time stages to solve intercity bus routing and scheduling problems under stochastic bus travel times.
Shen et al. (2016) assumed that trip times follow probability distributions, and then they devised a probabilistic model with the objectives of minimizing the total cost as well as maximizing the on-time performance.
Similar stochastic programming approaches can also be found in the area of vehicle routing problem (see, e.g., Pillac et al., 2013, for a comprehensive review).
However, many studies above fail to explicitly capture the delay propagation along adjacent service trips, which might make a significant impact on delay costs.
Furthermore, among the exiting studies concerning the delay propagation, few of them further investigates how to dynamically reschedule a vehicle fleet to reduce and control delay propagation effects.
Finally, it is still an open question whether large-scale realistic problems can be solved well.

### 2.2. Approximate dynamic programming

ADP emerges as a powerful tool for modeling and solving large and complex SDP.
Through introducing an approximated value function (cost-to-go function), ADP is capable of decomposing large-scale SDP.
Powell (2007) provides a comprehensive introduction of the basic ideas of ADP and addresses key algorithmic issues of ADP.
A large number of methods have been proposed in the ADP domain to solve realistic problems, such as adopting different value function approximation, updating rules as well as exploration versus exploitation strategies.
Interested readers are also referred to Bertsekas and Tsitsiklis (1996) and Sutton and Barto (2018).

Successful applications of ADP include transportation, finance, healthcare, energy, and supply chain management (e.g., Fang et al., 2013; Lei and Ouyang, 2017).
For instance, Papageorgiou et al.
(2014) considered a deterministic maritime inventory routing problem (MIRP), using the ADP method.
They used a value function approximation that is a separable piecewise linear continuous function and introduced a multi-period look-ahead strategy.
Rivera and Mes (2017) considered the planning problem faced by Logistic Service Providers, i.e., transporting freights periodically, and they used a ‘‘basis function” approach and the non-stationary least square method.
Yin et al.
(2016) considered a metro train rescheduling problem with uncertain time-variant passenger demands, and they utilized linear and separable basis functions.

Multilayered feed-forward neural networks perform as powerful and universal approximators for nonlinear mapping (Khosravi et al., 2011).
Due to the wide applicability, neural networks have been extensively used in the ADP and reinforcement learning fields as a powerful and adaptable class of nonlinear forms of value function approximations (Powell, 2007).
Neural networks can play an important role when it is difficult to find suitable basis functions to capture state features and there is no reasonable prediction of the nonlinear structures.
Specifically, the inputs to the network are the values of states and actions, and the outputs correspond to cost-togo function values.
This approach has been successfully applied in various fields.
Zhang et al. (2006) adopted a multi-layer feedforward neural network, trained with the levenberg-marquardt back-propagation algorithm in Q-learning, to help drivers choose the best route in complex traffic situations.
Miljković et al. (2013) used neural network reinforcement learning for the development and evaluation of visual control of robot manipulators.
Hajizadeh and Mahootchi (2016) used a radial-basis-function neural network to approximate the continuation value of the American option.

## 3.Model formulation

This section is devoted to formulating the stochastic dynamic VSP.
We begin with a description of the static VSP, using the connection-based model (e.g., Bunte and Kliewer, 2009).
We adopt a directed graph $J = ( N , A )$to represent the VSP, as shown in [Fig. 1](image/Fig1.png), where Zand Udenote the sets of nodes and links, respectively.
Considering the MDVSP, we denote the set of depots, where vehicles are dispatched, as $K$.
In the MDVSP, some trips have to be assigned to the vehicles from a certain set of depots (Löbel, 1998).
Therefore, there is a particular layer $J^k (N^k, A^k)$ of $J = (N,A)$ for each depot $k \in K$.
The node set $N^k$ kincludes the start depot $o^k$; the end depot $d^k$; the set of timetabled trips, i.e.$I^k$, each of which corresponds to a scheduled departure time and a stochastic trip time and must be fulfilled once by one bus; the set of virtual depots, i.e., $T^k$, which distinguish from the real depot nodes okand dkin the sense that buses returning to a virtual depot will later leave it and continue to fulfill service trips.
Note that the virtual depots are just used to accurately calculate the number of buses utilized, and they share the same physical locations as their corresponding real depots.
There are no buses at virtual depots at the beginning of an operations day.
Buses come to virtual depots after finishing some trips and later leave them to continue to fulfill other service trips.
In this context, to calculate the number of vehicles in a bus fleet, we only need to count vehicles departing from real depots.
Note that in the MDVSP, $o^k$ and $d^k$ share the same physical location, and we distinguish them to facilitate the formulation.
In order to always guarantee that the number of buses departing from virtual depots does not exceed the number of available buses, virtual depot nodes are time-expanded, i.e., different nodes in $T^k$ represent returning to the virtual depot at different times.
In reality, a bus can arrive at a virtual depot at any time, which results in an infinite number of virtual depot nodes in our network.
To diminish computational efforts, the time is sampled every $\beta$ minutes.

![Fig. 1. Illustrative example of a single-period VSP. ](image/Fig1.png)

![Fig. 2. Time horizon](image/Fig2.png)

The arc set $A^k$ includes the deadhead, pull-in and pull-out arcs.
Let $e_{ij}, (i,j) \in A^k$ represent the deadhead travel time from the end location of the trip ito the start location of the trip j.
aiand bidenote the start and end times of the trip i, respectively.
A schedule for a vehicle is corresponding to a path consisting of nodes and arcs.
Vehicles start and end at the same depot, and each trip of a given timetable is covered by exact one vehicle.
The vehicles’ schedules are optimized to minimize the total cost of arcs, including the travel cost (all arcs), waiting-time cost (deadhead arcs), delay cost (deadhead arcs) and fixed cost (pull-in arcs).
The waiting-time cost arises from the early arrival of a bus.
In this case, the bus has to wait at a bus station outside depots until its next trip starts.
In contrast, the delay cost results from the late arrival of a bus, and consequently, its next trip cannot start as scheduled.
It is worth mentioning that both waiting-time and delay costs have widely been considered in the literature of VSP (e.g., Huisman et al., 2004; Kliewer et al., 2006; Li, 2013).
The vehicle fixed cost represents the cost of including one additional vehicle in a bus fleet.
Note that when calculating the fixed cost, we do not count the vehicles departing from virtual depots, because these costs have been counted the first time these vehicles are used (in the pull-in arcs associated with the corresponding real depots).
As previously mentioned, on the days of operations and as time goes by, we observe actual arrival times of trips and can thus reschedule vehicles to fulfill subsequent trips, accordingly.
In the dynamic scheduling, we base on a finite planning horizon, which is divided into several periods.
At the beginning of each period, i.e.,$t=0,\dots,T$, we reschedule a bus fleet based on the current traffic condition.
With a little abuse of notation, we also refer to the period which begins at the time tas the period t, as shown in [Fig.2](image/Fig2.png).
We adopt the physical process in Powell (2007).
We will introduce the state and decision variables, exogenous information, and transition function before formulating the stochastic dynamic VSP.

### 3.1. State

The system state at time t, i.e., ${S_t} = (H_t,Q_t,P_t,G_t,R_t,U_t), t=0,\dots,T$, includes the trip sets $H_t$, the trip start time set $Q_t$, the trip end time set $G_t$, the set of depots 7twhere the assigned vehicles are originally dispatched, the set of the number of available vehicles at virtual depots $R_t$, and the sets of arcs $U_t$.
Now, we introduce ${S_t}$ in details.
Note that for notation simplicity, we omit the superscript kin [Sections 3.1–3.3](#31-state).
When scheduling at the time t, for the trips whose scheduled finish times are before the time $t+1$ , we need to determine their allocated vehicles’ next trips; otherwise, after fulfilling the trips, these vehicles will become idling until the time $t+1$.
Furthermore, for all the trips starting during the interval tand some urgent trips starting during the interval $t+1$ , we must assign them to buses at the time t, because waiting until the time $t+1$ to assign those trips to buses will incur a huge delay cost (even if a bus is directly dispatched from a depot).
To reflect these trips’ differences above, the trip sets at the time t, i.e., $H_t$, are divided into the following seven subsets

* $N^t_1$ : The set of trips starting before the time t, and already assigned to buses before the time t. The allocated buses’ next trips after fulfilling $N^t_1$ have not been determined up to the time t.
* $N^t_2$ : The set of trips beginning during the period t, and already assigned to buses before the time t. The scheduled trips’ finish times are after the time $t+1$.
* $\hat{N^t_2}$ : The set of trips beginning during the period t, and already assigned to buses before the time t. The scheduled trips’ finish times are before the time $t+1$.
* $N^t_3$ : The set of trips beginning during the period t, and not assigned to buses by the time t. The scheduled trips’ finish times are after the time $t+1$.
* $\hat{N^t_3}$ : The set of trips beginning during the period t, and not assigned to buses by the time t. The scheduled trips’ finish times are before the time $t+1$.
* $N^t_4$ : The set of trips beginning during the period $t+1$. They must be assigned to buses at the time t. Otherwise, if we wait until the time $t+1$ to assign those trips in $N^t_4$ to buses, a huge delay cost will be incurred even if a bus is directly dispatched from a depot.
* $N^t_5$ : The set of trips beginning during the period $t+1$. Those trips could be assigned to buses either at the time tor at $t+1$.

If we divide each trip node into two nodes that respectively represent the start and end of the corresponding trip, [Fig. 3](pics/Fig3.png) conceptually demonstrates the seven subsets above.

$Q_t$ and $P_t$ represent the sets of trips’ start and end times, respectively. [^1]
Specifically, we only consider the end times of the trips in$N^t_1$ and the start times of the trips in $N_{2}^{t}$ and $\widehat{N}_{2}^{t}$, i.e., $Q_{t}=\left\{a_{i}^{t} \mid i \in\left(N_{2}^{t} \cup \widehat{N}_{2}^{t}\right)\right\}$ and $P_{t}=\left\{b_{i}^{t} \mid i \in N_{1}^{t}\right\}$.
Note that this is attributed to the exogenous information we obtain at the time $t$, and we will explain the reason in Section 3.2.
$G_{t}$ is used to track the allocated vehicles' corresponding original depots and defined as $G_{t}=\left\{(i, k) \mid k \in K, i \in\left(N_{1}^{t} \cup N_{2}^{t} \cup \widehat{N}_{2}^{t}\right)\right\}$, where the trip $i$ was assigned to the vehicle originally departing from the depot $k$.
Set $R_{t}$ is $\left\{r_{i}^{t} \mid i \in T\right\}$ where $r_{i}^{t}$ represents the number of vehicles available for dispatching at the virtual depot $i$ at the time $t$.
Lastly, given the definition of the seven subsets of trips, based on the subsets to which an arc's start and end nodes belong, the arc set $U_{t}$ is divided into 11 subsets, which indicates possible deadhead trips in the current stage, i.e., $A_{1}^{t}=\left\{(i, j) \mid i \in N_{1}^{t}, j \in N_{3}^{t} \cup \widehat{N}_{3}^{t}\right\}, \quad A_{2}^{t}=\left\{(i, j) \mid i \in N_{1}^{t}, j \in N_{4}^{t}\right\}, \quad A_{3}^{t}=\left\{(i, j) \mid i \in N_{1}^{t}, j \in N_{5}^{t}\right\}, \quad A_{4}^{t}=\left\{(i, j) \mid i \in \widehat{N}_{2}^{t}, j \in N_{3}^{t} \cup \widehat{N}_{3}^{t}\right\}$, $A_{5}^{t}=\left\{(i, j) \mid i \in \widehat{N}_{2}^{t}, j \in N_{4}^{t}\right\}, \quad A_{6}^{t}=\left\{(i, j) \mid i \in \widehat{N}_{2}^{t}, j \in N_{5}^{t}\right\}, \quad A_{7}^{t}=\left\{(i, j) \mid i \in \widehat{N}_{3}^{t}, j \in N_{3}^{t} \cup \widehat{N}_{3}^{t}\right\}, A_{8}^{t}=\left\{(i, j) \mid i \in \widehat{N}_{3}^{t}, j \in N_{4}^{t}\right\}, \quad A_{9}^{t}=$ $\left\{(i, j) \mid i \in \widehat{N}_{3}^{t}, j \in N_{5}^{t}\right\}, A_{10}^{t}=\left\{(i, j) \mid i \in\left(N_{1}^{t} \cup \widehat{N}_{2}^{t} \cup \widehat{N}_{3}^{t}\right), j \in\{d\} \cup T\right\}$ and $A_{11}^{t}=\left\{(i, j) \mid i \in\{o\} \cup T, j \in\left(N_{4}^{t} \cup N_{5}^{t} \cup N_{3}^{t} \cup \widehat{N}_{3}^{t}\right)\right\}$

[^1]: Fig. 3 also demonstrates the difference between depots and virtual depots, i.e., there are no buses at virtual depots at the beginning of an operations day. Buses come to virtual depots after finishing some trips and later leave them to continue to fulfill other service trips.

### 3.2. Decision variables and exogenous information

On the day of operations, road traffic conditions could exhibit substantial uncertainty.
With the accumulation of historical operations dataset and the technology development of travel time prediction, it is reasonable to assume that at the time t, we are capable of accurately obtaining the information of the end times of all the ongoing trips, i.e., $Q_t$and $P_t$(Huisman et al., 2004; Yu et al., 2011; Xu and Ying, 2017).
However, for the trips that have not started yet, their travel times are stochastic variables, which equal $p_i=\bar{p_i} + f(a_i) + \Delta_i$ ,
where pi is the practical travel time for the trip i;
$\bar{p_i}$ represents the scheduled trip time on the timetable;
$\Delta_i$ is a stochastic variable, reflecting the uncertainty of road traffic conditions;
$f( a_i )$is a function to capture the influences of the start time delay of aion the trip i’s travel time.
Note that $f( a_i )$ could be substantial in urban transport where the line frequencies are high.
In an urban environment, if a trip starts later, its travel time could possibly be increased, because there are more passengers taking this trip who would have taken the next one if this trip had no delay (Huisman et al., 2004).
We further note that, for simplicity, we assume deadhead trips’ travel times to be constant.
It is straightforward to extend the proposed framework to consider the stochastic deadhead trip times.
To summarize, Table 1 shows the exogenous information (trip time information) we can obtain at the times tand $t+1$ , respectively.

**Table1** Exogenous information at the times t and $t+1$.

|      | t                   | t + 1                            |
| ---- | ------------------- | -------------------------------- |
| Nt1  | Start and end times | –                                |
| Nt2  | Start times         | End times                        |
| N t2 | Start times         | End times                        |
| Nt3  | –                   | Start and end times              |
| N t3 | –                   | Start and end times              |
| Nt4  | –                   | Start times                      |
| Nt5  | –                   | Start times if assigned to buses |

![Fig. 4. Transition of the trip sets](image/Fig4.png)

The decision made by public transport companies at the time $t$ can be denoted by ${X_t} = ( \cdots, x_{ij}^t, \cdots$,
where $x_{ij}^t$ is a binary variable, and it equals one if a vehicle is dispatched to fulfill the trip $j$ after finishing the trip $i$ and zero otherwise.

### 3.3. Transition function

Transitions include updating the trip and arc sets, and the information of trip time, the allocated vehicles’ depots and the number of available buses at virtual depots.
Specifically, at the time $t+1$ , the trips of $N^t_2$ and $N^t_3$ will be members of $N_1^{t+1}$ ; the trips of $N^t_4$ and the trips of $N^t_5$ , assigned to buses at the time t, will be members of $\hat{N_2}^{t+1}$ or $N_2^{t+1}$ ; the trips of $N^t_5$ , not assigned to buses at the time t, will be members of $N_3^{t+1}$ or $\hat{N_3}^{t+1}$.

[Fig. 4](image/Fig4.png) specifically demonstrates the transition of the trip sets.
After updating the trip set $H_{t+1}$ , the arc set $U_{t+1}$ can be accordingly defined, as per the specification in [Section 3.2](#32-decision-variables-and-exogenous-information).

With regard to how to obtain the trip time information in the column $t+1$ of Table 1, we first let $a_i^{t+1} = a_i^t, i \in N_2^t \cup \hat{N_2}^t, b_i^{t+1} = b_i^{t}, i \in N_1^t$;
and then we adopt the following two equations to update the rest trip time information.

$$
\begin{aligned}
&a_{i}^{t+1}=\max \left[\bar{a}_{i}, \sum_{(j, i) \in U^{t}} x_{j i}^{t}\left(b_{j}^{t+1}+e_{j i}\right)\right], i \in N_{3}^{t} \cup \widehat{N}_{3}^{t} \cup N_{4}^{t} \cup N_{5}^{t} \\
&b_{i}^{t+1}=a_{i}^{t+1}+\bar{p}_{i}+f\left(a_{i}^{t+1}\right)+\Delta_{i}, i \in N_{2}^{t} \cup \widehat{N}_{2}^{t} \cup N_{3}^{t} \cup \widehat{N}_{3}^{t}
\end{aligned}
$$

where $\bar{a}_{i}$ represents the trip $i$ 's scheduled start time on the timetable.
Following the two equations above, at the time $t+1$, we recursively update trips' start and end times along the scheduled paths of vehicles.
In this way, we are capable of explicitly considering delay propagation along different trips fulfilled by the same vehicle.
After updating the corresponding trip time information, we can obtain $Q_{t+1}$ and $P_{t+1}$, based on the transition of trip sets, as shown in Fig. 4.
To update $G_{t+1}$, we need to first introduce an auxiliary set $\widetilde{G}_{t}$.
Specifically, let $\widetilde{G}_{t}=G_{t}$, and we recursively update $\widetilde{G}_{t}$ through $\widetilde{G}_{t}=\widetilde{G}_{t} \cup\left\{\left(i, \sum_{k \in K,(, k) \in \widetilde{G}_{t}} \sum_{(j, i) \in U^{t}} x_{j i}^{t} k\right)\right\}$.
Note that this update starts from the trips in the sets $\left(N_{1}^{t} \cup N_{2}^{t} \cup \widehat{N}_{2}^{t}\right)$ and continues along the scheduled paths of vehicles.
After finishing updating $\widetilde{G}_{t}$, we are capable of identifying the sets $\left\{(i, k) \mid k \in K, i \in\left(N_{1}^{t+1} \cup N_{2}^{t+1} \cup \widehat{N}_{2}^{t+1}\right)\right\}$, i.e., $G_{t+1}$, from $\widetilde{G}_{t}$, based on the transition relationship of trip sets, as shown in Fig. 4. With regard to $R_{t+1}$, we update it through $r_{i}^{t+1}=r_{i}^{t}+\sum_{j,(, i) \in U_{t}} x_{j i}^{t}-\sum_{j,(i, j) \in U_{t}} x_{i j}^{t}$.
Finally, to better illustrate the transition function, we provide a demonstrative example in the Appendix.

### 3.4. Dynamic programming formulation

The objective of dynamic scheduling is to minimize the total costs, which consist of the deadhead travel time costs, waiting-time costs, vehicle fixed costs, and delay costs. [^2]
Let $c_{d}$ be the cost of unit travel distance, $\bar{d}_{i j}$ be the distance of the arc $(i, j)$, i.e., the deadhead trip $(i, j), c_{b}$ be the fixed vehicle cost, and $c_{w}$ be the cost of unit waiting-time.
Similar to Huisman et al. (2004) and to reflect the huge adverse impacts of long delays, we adopt a quadratic penalty function to calculate delay costs.
Specifically, suppose that the cost of a delay of $\alpha$ minutes is the same as the fixed vehicle cost $c_{b}$, and then the cost associated with a $r$-minute start delay of a trip is calculated as $\left(r^{2} / \alpha^{2}\right) \cdot c_{b}$.
If we denote the cost in the period $t$ as $C_{t}\left(\boldsymbol{S}_{t}, \boldsymbol{X}_{t}\right)$, we can calculate it through $C_{t}\left(\boldsymbol{S}_{t}, \boldsymbol{X}_{t}\right)=c_{b} z^{t}+\sum_{k \in K} \sum_{(i j) \in U_{t}^{k}} x_{i j}^{k, t} c_{d} \bar{d}_{i j}+\sum_{i \in\left(N_{3}^{t-1} \cup \widehat{N}_{3}^{t-1} \cup N_{2}^{t} \cup \hat{N}_{2}^{t}\right)} \widetilde{c}_{i}^{t}+\sum_{i \in\left(N_{3}^{t-1} \cup \hat{N}_{3}^{t-1} \cup N_{2}^{t} \cup \hat{N}_{2}^{t}\right)} \widetilde{w}_{i}^{t}$, where $z^{t}$ represents the number of vehicles newly used during the period t ;
$\widetilde{c}_{i}^{t}$ denotes the delay cost of the trip $i$, and $\widetilde{w}_{i}^{t}$ is the waiting-time cost of the trip $i$.
Note that at the time $t$, based on Table 1 , we only obtain the information of the start times of the trips in the sets $\left(N_{3}^{t-1} \cup \widehat{N}_{3}^{t-1} \cup N_{2}^{t} \cup \widehat{N}_{2}^{t}\right)$, which explains why the delay and waiting-time costs in $C_{t}\left(\boldsymbol{S}_{t}, \boldsymbol{X}_{t}\right)$ only consider the trips in these sets.
Under the notation above, the finite-time-horizon stochastic VSP (S-VSP) can be formulated as below.

[^2]: Note that the total costs do not include the service trip time costs in the consideration of the fact that the service trip time is independent of the scheduling optimization except for the impact of delay, i.e., $f( a_i )$. In fact, this impact of delay on the service trip time is already incorporated into delay costs.

$$
\begin{aligned}
&\min _{\boldsymbol{X}_{t}}\left\{\sum_{t=0}^{T} C_{t}\left(\boldsymbol{S}_{t}, \boldsymbol{X}_{t}\right)\right\} \\
&\sum_{j,(i, j) \in U_{t}^{k}} x_{i j}^{k, t}=1 \quad \forall i \in N_{1}^{t} \cup \widehat{N}_{2}^{t}, k \in K,(i, k) \in G_{t}, t \in\{0, \cdots, T\} \\
&\sum_{k} \sum_{i,(i)) \in U_{t}^{k}} x_{i j}^{k, t}=1 \quad \forall j \in \widehat{N}_{3}^{t} \cup N_{3}^{t} \cup N_{4}^{t}, t \in\{0, \cdots, T\} \\
&\sum_{k} \sum_{i,(i, j) \in U_{t}^{k}} x_{i j}^{k, t} \leq 1 \quad \forall j \in N_{5}^{t}, t \in\{0, \cdots, T\} \\
&\sum_{j,(, i) \in U_{t}^{k}} x_{j i}^{k, t}-\sum_{j,(i j) \in U_{t}^{k}} x_{i j}^{k, t}=0, \quad \forall i \in \widehat{N}_{3}^{t}, k \in K, t \in\{0, \cdots, T\} \\
&r_{i}^{t}+\sum_{j,(, i) \in \cup U_{t}^{k}} x_{j i}^{k, t}-\sum_{j,(i, j) \in U_{t}^{k}} x_{i j}^{k, t} \geq 0 \quad \forall k \in K, i \in T^{k}, t \in\{0, \cdots, T\} \\
&z^{t}=\sum_{k} \sum_{j,\left(0^{k}, j\right) \in U_{t}^{k}} x_{o^{k} j}^{k, t} \quad t \in\{0, \cdots, T\} \\
&x_{i j}^{k, t}=\{0,1\} \quad \forall i j \in U_{t}^{k}, k \in k, t \in\{0, \cdots, T\}
\end{aligned}
$$

As formulated, the objective function is to minimize the expectation of the total costs within the entire planning time horizon.
Constraint (1) guarantees that the vehicles will not stop after finishing the trips in $N_{1}^{t}$ and $\widehat{N}_{2}^{t}$ since these trips end before the time $t+1$.
Constraint (2) makes sure that the trips of $\widehat{N}_{3}^{t}, N_{3}^{t}$ and $N_{4}^{t}$ are assigned to vehicles at the time $t$ because waiting until the time $t+1$ will incur huge delays.
Constraint (3) means that the corresponding arcs can be determined either at the time $t$ or at $t+1$.
Constraint (4) is the flow conservation constraint and constructs coherent paths for depots.
Constraint (5) indicates that the number of dispatched vehicles from a virtual depot cannot exceed its remaining vehicle number.
Constraint (6) calculates the number of vehicles newly utilized during the period $t$.
Lastly, constraint (7) requires $x_{i j}^{k, t}$ to be binary.

## 4. ADP algorithm development

Using the post-decision state variable, i.e., the state of the system after making a decision but before any new information has arrived, denoted by • ta, we can write the Bellman’s equation as follows.

$$
V_{t}\left(\boldsymbol{S}_{t}\right)=\min _{\boldsymbol{X}_{\boldsymbol{t}}}\left[C_{t}\left(\boldsymbol{S}_{\boldsymbol{t}}, \boldsymbol{X}_{\boldsymbol{t}}\right)+V_{t}^{a}\left(\boldsymbol{S}_{\boldsymbol{t}}^{\boldsymbol{a}}\right)\right]
$$

s.t. Constraints (1) - (7)

where $V_{t}^{a}\left(\boldsymbol{S}_{t}^{a}\right)=E\left\{V_{t+1}\left(\boldsymbol{S}_{t+1}\right) \mid \boldsymbol{S}_{t}, \boldsymbol{X}_{t}\right\}$ is the expected value of being in the state $\boldsymbol{S}_{t}^{a}$ immediately after we made a decision, i.e., cost-togo function.
The S-VSP can be solved using the classical backward dynamic programming algorithm by recursively solving the Bellman's equation.
However, the backward dynamic programming faces the curses of dimensionality and can turn out to be computationally intractable.
For instance, if there are 500 trips, 2 depots and 9 periods, there are roughly 2000 possible arcs every time we make a decision.
Since the decision variable is binary, the action space might have $2^{18000}$ outcomes.
Furthermore, when delay propagation is considered, the state and outcome spaces may be even larger. To tackle these challenges, we adopt an ADP approach, which is a powerful tool for solving large scale dynamic programs.

In an ADP approach, instead of stepping backward in time, we use the approximated value function to replace the value function $V_t^a(S_t^a)$ in the Bellman’s equation and step forward in time to make decisions.
The next state is a random variable dependent on current decisions and sample paths, and it is determined based on the transition function with the observed stochastic information.
The value function approximation is updated using the realized values of visited states as the algorithm progresses.
Such procedure runs iteratively, using newly sampled realization each time, and it gradually reaches more accurate approximated value functions and better policies.

In the following subsections, we analyze the composition of a cost-to-go function first and then propose the value function approximations using neural networks.
Then, we use the stochastic gradient method to update the approximations following the TD $(\lambda)$ procedure.
In order to improve the performance of the ADP approach, some key algorithmic issues are discussed later, such as the step size rules and exploration *vs.* exploitation.

### 4.1. Value function approximations

$V_{t}^{a}\left(\boldsymbol{S}_{t}^{a}\right)$ is the optimal expected future cost of being in state $\boldsymbol{S}_{t}$ and taking an action $\boldsymbol{X}_{t}$.
As mentioned earlier, in an ADP approach, we propose a function $\bar{V}_{t}^{a}\left(\boldsymbol{S}_{t}^{a}\right)$ to approximate the value function $V_{t}^{a}\left(\boldsymbol{S}_{t}^{a}\right)$.
Considering that the decision variable $\boldsymbol{X}_{t}$ is binary, a linear or piecewise linear function form might not be appropriate for $\bar{V}_{t}^{a}\left(\boldsymbol{S}_{t}^{a}\right)$.
Specifically, it is difficult to construct post-decision variables $\boldsymbol{S}_{t}^{a}$, using vehicle dispatching plans $\boldsymbol{X}_{t}$ and trip time information in $\boldsymbol{S}_{t}$; so we use state-action pairs instead.
In other words, the inputs of $V_{t}^{a}\left(\boldsymbol{S}_{t}^{a}\right)$ contain both binary decisions and trip time information, which means that we need to find proper $\bar{V}_{t}^{a}\left(\boldsymbol{S}_{t}^{a}\right)$ with the inputs of both binary variables and continuous variables.
This differs from many previous studies adopting piecewise-linear approximate value functions (e.g., Godfrey and Powell, 2001, 2002).

![Fig. 5. Three-layer neural network for the approximated value function](image/Fig5.png)

To enhance the accuracy, we consider more complicated value function approximations. Neural networks, widely used in ADP as a powerful and adaptable class of nonlinear forms of value function approximations, offer much more flexible groups of frameworks and can also be updated recursively (e.g., Zhang et al., 2006; Powell, 2007; Hajizadeh and Mahootchi, 2016). In this study, we use a three-layer feed-forward neural network, as shown in Fig. 5, to approximate $V_{t}^{a}\left(\boldsymbol{S}_{t}^{a}\right)$. The neurons receiving in the three-layer feedforward neural network's input layer include: (i) the decision variable $\boldsymbol{X}_{t}$; (ii) the state variable $P_{t}$, i.e., $\left(b_{i}^{t}-t \cdot L\right.$ ) $/ L, i \in N_{1}^{t}$; (iii) the state variable $Q_{t}$, i.e., $\left(a_{i}^{t}-t \cdot L\right) / L, i \in N_{2}^{t} \cup \widehat{N}_{2}^{t}$, where $L$ represents the duration of one time period. The output neuron is the estimated value of the value function $V_{t}^{a}\left(\boldsymbol{S}_{t}^{a}\right) .$ Relu function is utilized to activate the perceptron nodes.

Let $F_{q}^{l, t}$ represent the input unit $q$ in the layer $l$ of the neural network and $\Gamma^{l}$ denote the set of input units in the layer $l$ of the neural network. As mentioned earlier, the vector $\left(\cdots F_{q}^{1, t}, \cdots\right), q \in \Gamma^{1}$, includes the three kinds of neurons receiving above. The adopted threelayer feed-forward neural network is characterized as follows.

$$
\begin{aligned}
&Y_{q}^{2, t}=\sum_{q \in \Gamma^{1}} W_{q, q}^{1, t} \cdot F_{q}^{1, t}+v_{q}^{1, t} \quad \forall q \in \Gamma^{2} \\
&F_{q}^{2, t}=\max \left(0, Y_{q}^{2, t}\right) \quad \forall q \in \Gamma^{2} \\
&\nabla_{t}^{a}\left(\boldsymbol{S}_{t}^{a}\right)=\sum_{q \in \Gamma^{2}} W_{q}^{2, t} F_{q}^{2, t}+v^{2, t}
\end{aligned}
$$

where $W_{q, q}^{1, t}$ indicates the weight between the unit $q$ in the layer 1 and the unit $q$ in the layer $2 ; W_{q}^{2, t}$ is the weight associated with the input unit $q$ in the layer $2 ; v^{1, t}$ and $v^{2, t}$ are the bias of the unit $q$ in layer 1 and the bias in layer 2 , respectively. Constraint (9) uses the Relu function to activate $Y_{q}^{2, t}$. Constraint (10) defines the approximated value function. Note that the parameters $W_{q, q}^{1, t}, W_{q}^{2, t}, v_{q}^{1, t}$ and $v^{2, t}$ will be iteratively updated using the realized values of visited states, as the ADP algorithm progresses. According to the neuralnetwork-based approximated functions above, the Bellman's equation can be reformulated as follows.

$$
V_{t}\left(\boldsymbol{S}_{t}\right)=\min _{\boldsymbol{X}_{t}}\left[C_{t}\left(\boldsymbol{S}_{t}, \boldsymbol{X}_{t}\right)+\bar{V}_{t}^{a}\left(\boldsymbol{S}_{t}^{a}\right)\right]
$$

s.t. Constraints (1) - (10)
In this way, we circumvent the embedded expectation in the Bellman's equation. Note that it is straightforward to reformulate the model above to be a mixed integer linear program, which we don't present for brevity.

### 4.2. Value function updating

Define $\boldsymbol{\Theta}=\left(\cdots, W_{q_{1}, q_{2}}^{1, t}, \cdots, W_{q_{3}}^{2, t}, \cdots, v_{q_{4}}^{1, t}, \cdots v^{2, t}, \cdots\right), \forall q_{1} \in \Gamma^{1}, q_{2}, q_{3}, q_{4} \in \Gamma^{2}$.
We adopt the stochastic gradient algorithm to update $\boldsymbol{\Theta}$ (e.g., Powell, 2007; Fang et al., 2013), which is shown as follows.

$$
\Theta^{m}=\Theta^{m-1}-\alpha_{m-1}\left(\bar{V}_{t}^{a, m-1}\left(\boldsymbol{S}_{t}^{a}\right)-\widehat{v}_{t+1}\right) \nabla_{\Theta} \bar{V}_{t}^{a, m-1}\left(\boldsymbol{S}_{t}^{a}\right)
$$

where the superscript $m$ represents the iteration number; $\alpha_{m-1}$ is the step size at iteration $m-1 ; \hat{v}_{t+1}$ is the realized value resulting from the current policy and sample path; $\nabla_{\Theta} \bar{V}_{t}^{a, m-1}\left(\boldsymbol{S}_{t}^{a}\right)$ represents the gradient of the approximated value function with respect to the parameter vector $\boldsymbol{\Theta}$ at the iteration $m-1$.
Specifically, $\nabla_{\Theta} \bar{V}_{t}^{a, m-1}\left(\boldsymbol{S}_{t}^{a}\right)$ can be calculated based on Eqs. (8)-(10) using the chain rule. [^3] With regard to the calculation of $\left(\bar{V}_{t}^{a, m-1}\left(\boldsymbol{S}_{t}^{a}\right)-\hat{v}_{t+1}\right)$, we adopt the temporal difference method, which is shown as below.

[^3]: Note that because function $F_{q}^{2, t}$ is not differentiable at $Y_{q}^{2, t}=0 .$ Hence, we use its subgradient to facilitate the calculation of $\nabla_{\Theta} \bar{V}_{t}^{a, m-1}\left(\boldsymbol{S}_{t}^{a}\right)$.

$$
\hat{v}_{t}=\bar{V}_{t-1}^{a, m-1}\left(\boldsymbol{S}_{t-1}^{a}\right)+\sum_{\tau=t}^{T} \lambda^{\tau-t}\left[C_{\tau}\left(\boldsymbol{S}_{\tau}^{m-1}, \boldsymbol{X}_{\tau}^{m-1}\right)+\bar{V}_{\tau}^{a, m-1}\left(\boldsymbol{S}_{\tau}^{a}\right)-\bar{V}_{\tau-1}^{a, m-1}\left(\boldsymbol{S}_{\tau-1}^{a}\right)\right]
$$

The equation above produces the $\operatorname{TD}(\lambda)$ updating method, widely adopted in the ADP field. If we consider $\left[C_{\tau}\left(\boldsymbol{S}_{\tau}^{m-1}, \boldsymbol{X}_{\tau}^{m-1}\right)+\bar{V}_{\tau}^{a, m-1}\left(\boldsymbol{S}_{\tau}^{a}\right)-\bar{V}_{\tau-1}^{a, m-1}\left(\boldsymbol{S}_{\tau-1}^{a}\right)\right]$ as a correction to the estimate of value function, the discounting factor $0 \leq \lambda \leq 1$ is utilized to reflect that updates farther along the sample path should not be given so much weight as those earlier in the path.

### 4.3. Step-size rule

 An important issue in the ADP algorithm is choosing a proper step-size rule because of its significant impact on the convergence and solution performance.
Due to the lack of change with various realized and collected data, deterministic step-size rules may possibly lead to a slow rate of convergence.
In this study, we adopt an optimal stochastic step-size rule for non-stationary data with a steadily increasing or decreasing mean, i.e., the Bias-adjusted Kalman Filter (BAKF) step-size rule (e.g., George and Powell, 2006; Powell, 2007; Fang et al., 2013), which is given by.

$$
\alpha_{m-1}=1-\frac{\bar{\sigma}^{2}}{\left(1+\varkappa_{m-1}\right) \bar{\sigma}^{2}+\left(\eta_{m}\right)^{2}}
$$

where $\bar{\sigma}^{2}$ indicates the estimate of the variance of $\bar{V}_{t}^{a, m}\left(\boldsymbol{S}_{t}^{a}\right)$ and $\eta_{m}$ represents the estimate of bias for using simple smoothing on nonstationary data; lastly, $x_{m-1}$ is computed using $\chi_{m}=\left\{\begin{array}{cc}\left(\alpha_{m-1}\right)^{2} & m=1 \\ \left(1-\alpha_{m-1}\right)^{2} \kappa_{m-1}+\left(\alpha_{m-1}\right)^{2} & \mathrm{~m}>1\end{array} .\right.$ For more details, refer to Section $11.4$ in Powell (2007).

### 4.4. Exploration vs. exploitation

A modified mixed exploration strategy is used since we may ignore some states and actions that may not seem attractive when we have not visited them often enough.
The rate of exploration is set to be a piecewise linear function of the iteration number m.
At early iterations, we explore more states randomly to update the inaccurate initialization of value function approximations, whereas at latter iterations we exploit the current estimates of state values to obtain a decision that we think is the best.
In the numerical study section, we will study the performance of various exploration rates.
Finally, the detailed procedure of the proposed ADP algorithmic framework is shown as below.

![ADP algorithmic framework](image/ADP_algorithmic_framework.png)

## 5. Numerical examples

 In this section, we conduct the numerical examples based on the realistic operations dataset of bus lines in Beijing.
Specifically, we extract the realized travel times of 520 trips that repeat every day for 11 bus lines and two depots in Beijing between October 1st and 15th of 2015.
The first trip starts at 7:00 and the last trip starts at 22:00.
Based on the dataset’s service trips’ realized travel times, we set the duration of each decision period to be 110 min,[^4] and the planning horizon is accordingly divided into nine periods.
We use the myopic and static policies as benchmarks.
The myopic policy does not take into account future impacts and makes short-sighted decisions only based on the current-period cost.
The static policy optimizes a bus fleet’s schedule of the entire planning horizon ahead of the start of an operations day and implements the obtained schedule regardless of the bus fleet’s operations condition during the day.
The algorithms are programmed in C++ with Microsoft Visual Studio 2012.
All the computational experiments are implemented on a computer with Intel(R) Core(TM) i5-4460 CPU@3.20 GHz and 16 GB RAM.
To mimic the stochasticity of trip times, we first calculate the average value of each trip’s duration based on the realistic dataset, and then it is assumed that the realized travel time of a trip follows a truncated normal distribution ranging from 60% to 140% of its calculated average value, with the standard deviation equal to 20% of its average value.
Lastly, unless otherwise specified, all the other parameter values are set based on the specification by Huisman et al.
(2004).

[^4]: In the dynamic scheduling process, the suitable duration of one period should be longer than one trip’s travel time and shorter than the total travel time of two consecutive trips. If the duration is too long, we are not capable of adjusting schedules in time. If the duration is too short and rescheduling happens too often, its effect is limited. Finally, we note that choosing 110 min also takes into account the factor of avoiding many trips’ start times.

### 5.1. Convergences

Using the BAKF stepsize rule [^5], and setting $\rho=0.1$ for the first 200 iterations and $\lambda=0.2$, we plot the convergence curve of the ADP method in the case of 520 trips in Fig. 6 .
The average calculation time for every iteration is $23.26 \mathrm{~s}$ and it takes about $4652.66 \mathrm{~s}$ to converge.
In Fig. 6, the top curve depicts the realized total costs under different iterations numbers; the middle curve shows the change of the total realized cost excluding the cost of the first period; the bottom curve represents the values of $\bar{V}_{0}^{a}\left(\boldsymbol{S}_{0}^{a}\right)$ under different iteration numbers. It can be observed that as the iteration number increases, the approximate value function $\bar{V}_{0}^{a}\left(\boldsymbol{S}_{0}^{a}\right)$ gradually approaches the realized total costs excluding the cost of the first period, and the total realized cost reduces, which demonstrates the effectiveness of our proposed ADP approach for dynamically scheduling the vehicle fleet under stochastic trip times.
Note that the bottom curve approaches the middle curve instead of the top curve because we use the post-decision states in the ADP approach and $\bar{V}_{0}^{a}\left(\boldsymbol{S}_{0}^{a}\right)$ is an approximate for the value function from the second period to the end of the operations horizon.

### 5.2. Comparisons of ADP and benchmark policies

To compare the ADP approach with the static and myopic policies, we further extract 87 trips from 520 trips so that the connection-based model under the static policy, i.e., MDVSP, can be directly and exactly solved by commercial solvers such as CPLEX.
We first simulate 150 times for the myopic, the static and the ADP policies.
The ADP policy uses the approximated value function obtained after convergence of iterations, and we set 0 0.1and 0 0.4.
For the static policy, we introduce a buffer time of 30 min so that delay propagation will not lead to an extremely large delay cost.
Table 2 compares the average values of realized total costs under these three policies under different values of the relative standard deviation.
The relative standard deviation is corresponding to the distributions of stochastic trip times; for instance, 0.2 means that the standard deviation value is equal to 20% of the average value.
It can be observed that as the trip times’ stochasticity increases, the advantage of the ADP method over both benchmark policies becomes more significant.
Figs.
7 and 8 respectively depict the bus fleet schedules derived under the ADP and myopic policies in one simulation, with a relative standard deviation of 0.2.
Examining the details of these bus fleet schedules, we can summarize some reasons why the ADP policy outperforms the myopic policy.

* The myopic policy chooses actions just based on the costs of current period. It often fails considering the travel costs from virtual     depots to trips when rescheduling, because these costs may not belong to current periods. In consequence, it is observed that the  myopic policy sends buses to return to virtual depots too frequently, as shown in Fig.8.
* The ADP policy considers potential delay costs in future periods and thus chooses relatively conservative deadhead trips, i.e., the     slopes of some dotted lines in Fig. 7 are lower than those in Fig.1.
* In the MDVSP, some trips starting in latter periods can only be fulfilled by buses from particular depots. The myopic policy fails taking into account this factor when scheduling in early periods, because it only focuses on the operations of current periods. Consequently, some buses dispatched in early periods face a relatively low level of utilization in latter periods, because the depot     restriction prevents them from serving many trips, which leads to a waste of resources. Note that this point cannot directly be observed from comparing Figs. 7 and 8, but examining the detailed schedules can reveal it.

Table 3 compares the average realized operations and delay costs of the static and ADP models, with the relative standard deviation as 0.2.
We observe that, the advantage of ADP, compared to the static policy, is mainly attributed to its ability of significantly reducing the delay costs.

### 5.3. Exploration vs. exploitation

In the proposed ADP algorithm, we randomly choose an action with a fixed probability at the first 200 iterations.
After it, we stop the exploration and choose the exact actions that minimize the total costs based on our learned value function.
We compare the performances of the ADP algorithm with various values of in Table 4. Fig. 9 shows their convergence curves.
In Table 4, we can observe that the ADP algorithm with = 0.1 balances the exploration and exploitation well, yielding a better solution and minor fluctuation after convergence.
The possible explanation is that when the exploration rate is too low, it takes a shorter time for value functions to converge but the obtained policy is not efficient enough because of ignoring many possible states; when the exploration rate is too high, many unnecessary states are explored, causing the difficulties for value functions to converge.

[^5]: We use the harmonic stepsize rule at early iterations, when we cannot calculate the BAKF stepsize accurately, as introduced in Powell (2007), to reduce calculation times.

![Fig.6. Convergence curves of ADP using neural networks.](image/fig6_s.png)

**Table 2** Comparisons of the average realized total costs under the ADP and benchmark policies

| Relative standard deviation | ADP       | Static    | Myopic    |
| --------------------------- | --------- | --------- | --------- |
| 0.1                         | 111972.71 | 121037.94 | 118743.03 |
| 0.2                         | 121602.28 | 159911.86 | 129679.47 |
| 0.3                         | 124303.41 | 192940.28 | 135789.57 |

[Fig.7. Time-space diagram of the schedule under the ADP policy.](pics/Fig7.png)

[Fig.8.Time-space diagram of the schedule under the myopic policy.](pics/Fig8.png)

**Table 4** Performances of different ρ.

|                    | ρ= 0      | ρ= 0.1    | ρ= 0.2    | ρ= 0.3    |
| ------------------ | --------- | --------- | --------- | --------- |
| Mean value         | 610821.15 | 606602.90 | 618326.46 | 619443.61 |
| Standard deviation | 28828.24  | 27088.33  | 30863.99  | 28891.38  |

![Fig.9. Convergence curves under different rates of ρ.](image/fig9_s.png)

![Fig.10. Convergence curves under different values of λ.](image/fig10_s.png)

**Table 5** The performances of different .

|                    | λ= 0      | λ= 0.2    | λ = 0.4   | λ = 0.6   |
| ------------------ | --------- | --------- | --------- | --------- |
| Mean value         | 624124.23 | 606602.90 | 617035.21 | 619894.42 |
| Standard deviation | 35089.91  | 27088.33  | 31013.16  | 28779.52  |

### 5.4. The influence of λ

Recall that we update the value function, i.e., the neural network, through the method of $TD(\lambda)$.
The performance of $TD(\lambda)$ is affected by the value of discounting factor.
The convergence curves under different values of are depicted in Fig.
10 and their performances are compared in Table 5.
In Table 5, it is obvious that the $TD(\lambda)$ method with = 0.2 provides a better solution and minor fluctuation after convergence.
In fact, if the value of is too high, considering the impacts of current actions on future total costs too much may mislead the policy because approximate value functions are not accurate enough at early iterations.
If the value of is too low, approximate value functions converge slowly since we do not pay enough attention to future costs.
Finally, to let the readers better understand Figs.
9 and 10, we calculate the average values of realized total costs for every 50 iterations and plot the convergence trends based on the calculated average values in the Appendix.

## 6. Conclusions

In this paper, we present the ADP approach to solve the MDVSP under stochastic travel times with the full consideration of delay propagation effects.
The proposed framework dynamically guides the operations of a bus fleet to cope with complex traffic conditions and comprehensively consider both the operations costs and punctuality of the transit system.
In particular, a three-layer feedforward neural network is utilized to approximate a value function so that the problem size and complexity can be effectively reduced and we are capable of stepping forward to make decisions and solving the Bellman’s equation through sequentially solving multiple MIP problems.
As the ADP algorithm progresses, the value function approximation is updated using the realized values of visited states and following the stochastic gradient algorithm and $TD(\lambda)$ method.
Numerical examples based on the realistic operations dataset of bus lines in Beijing have demonstrated that our neural-network-based ADP approach not only exhibits a good learning behavior but also performs better in solving large-scale practical problems compared to both myopic and static polices, especially when trip time stochasticity is high.
Future research will focus on extending the proposed framework to consider the scheduling problem of alternative fuel vehicles under stochastic traffic conditions.
Compared to the VSP of conventionally-fueled diesel buses, the alternative-fuel VSP needs to additionally take into account range limitation and recharging plans, which will inevitably increase the model complexity and bring computational challenge (e.g., Chao and Chen, 2013; Adler et al., 2016).

## Acknowledgements

The research is partially supported by grants from National Natural Science Foundation of China (51622807, 71501107, U1766205).
The research is supported in part by the Center for Data-Centric Management in the Department of Industrial Engineering at Tsinghua University.
The authors would like to thank the anonymous reviewers for their helpful comments and suggestions.

## Appendix A

In this appendix, we first provide a simple example to illustrate the transition function, as shown in Fig. A1.
The trip time

[Fig.A1.Simple example to illustrate the transition function.](pics/FigA1.png)

**Table A1** Trip time information.

| Trip | Scheduled start time | Scheduled end time | Random variable (mins) |
| ---- | -------------------- | ------------------ | ---------------------- |
| i    | 8:00                 | 8:50               | 25                     |
| j    | 9:10                 | 10:00              | -10                    |
| k    | 11:10                | 12:00              | 10                     |

information is shown in Table A1, where the trips' scheduled start and end times are predefined, and the values of stochastic variables, i.e., $\Delta_{i}$, are gradually observed, as demonstrated in Table 1 . Suppose that the deadhead travel times between these trips are all set to be $20 \mathrm{~min}$, and it takes $25 \mathrm{~min}$ for buses to travel from depots to the start stations of the trip $j$ or $k$. The times $t-1, t$, and $t+1$ correspond to 7:10, 9:00, and 10:50, respectively.

When we schedule trips at the time $t-1$, the trip $j$ belongs to the set $N_{4}^{t-1}$ because there is no enough time for a bus to arrive at 9:10 to fulfill it if we assign it to the bus at the time $t$. Suppose that at the time $t-1$, we dispatch a bus to first fulfill the trip $i$ and then the trip $j$. As time goes by, we observe the value of $\Delta_{i}$, and that the trip $i$ starts on time. At the time $t$, the trip $i$ 's end time is updated using $b_{i}^{t}=a_{i}^{t}+\bar{p}_{i}+f\left(a_{i}^{t}\right)+\Delta_{i}$, where $f\left(a_{i}^{t}\right)=0$ because the trip $i$ starts on time, and we have $b_{i}^{t}=9: 15$. The trip $j$ 's start time is then updated using $a_{j}^{t}=\max \left\{9: 10, b_{i}^{t}+20\right.$ mins $\}=9: 35$. Assume that $f\left(a_{j}^{t}\right)=0.2\left(a_{j}^{t}-\bar{a}_{j}\right)$, which means that a start delay of 25 min will cause an additional travel time of five minutes. When we schedule trips at the time $t$, the trip $j$ belongs to $\widehat{N}_{2}^{t}$, and suppose that we dispatch the bus to fulfill the trip $k$ after it finishes fulfilling the trip $j$. At the time $t+1$, the trip $j$ 's end time is updated using $b_{j}^{t+1}=9: 35+(5+40) \operatorname{mins}=10: 20$. Similarly, $a_{k}^{t+1}=\max \left(11: 10, b_{j}^{t+1}+20\right.$ mins $)=11: 10$. Finally, the trips $j$ 's and $k$ 's corresponding depots are updated to be the same one as the trip $i$ 's.

Next, to let the readers better understand Figs. 9 and 10, we calculate the average values of realized total costs for every 50 iterations and plot the convergence trends based on the calculated average values in Figs. A2 and A3.

![Fig. A2. Convergence trends under different rates of ρ.](image/figA2_s.png)

![Fig. A3. Convergence trends under different values of λ.](image/figA3_s.png)

## References

removed
