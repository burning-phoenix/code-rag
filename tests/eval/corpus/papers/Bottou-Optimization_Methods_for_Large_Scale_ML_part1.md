

---

## Page 1

# Optimization Methods for Large-Scale Machine Learning

Léon Bottou\*

Frank E. Curtis<sup>†</sup>Jorge Nocedal<sup>‡</sup>

February 12, 2018

## Abstract

This paper provides a review and commentary on the past, present, and future of numerical optimization algorithms in the context of machine learning applications. Through case studies on text classification and the training of deep neural networks, we discuss how optimization problems arise in machine learning and what makes them challenging. A major theme of our study is that large-scale machine learning represents a distinctive setting in which the stochastic gradient (SG) method has traditionally played a central role while conventional gradient-based nonlinear optimization techniques typically falter. Based on this viewpoint, we present a comprehensive theory of a straightforward, yet versatile SG algorithm, discuss its practical behavior, and highlight opportunities for designing algorithms with improved performance. This leads to a discussion about the next generation of optimization methods for large-scale machine learning, including an investigation of two main streams of research on techniques that diminish noise in the stochastic directions and methods that make use of second-order derivative approximations.

## Contents

<table>
<tr>
<td><b>1</b></td>
<td><b>Introduction</b></td>
<td><b>3</b></td>
</tr>
<tr>
<td><b>2</b></td>
<td><b>Machine Learning Case Studies</b></td>
<td><b>4</b></td>
</tr>
<tr>
<td>2.1</td>
<td>Text Classification via Convex Optimization . . . . .</td>
<td>4</td>
</tr>
<tr>
<td>2.2</td>
<td>Perceptual Tasks via Deep Neural Networks . . . . .</td>
<td>6</td>
</tr>
<tr>
<td>2.3</td>
<td>Formal Machine Learning Procedure . . . . .</td>
<td>9</td>
</tr>
<tr>
<td><b>3</b></td>
<td><b>Overview of Optimization Methods</b></td>
<td><b>13</b></td>
</tr>
<tr>
<td>3.1</td>
<td>Formal Optimization Problem Statements . . . . .</td>
<td>13</td>
</tr>
<tr>
<td>3.2</td>
<td>Stochastic vs. Batch Optimization Methods . . . . .</td>
<td>15</td>
</tr>
<tr>
<td>3.3</td>
<td>Motivation for Stochastic Methods . . . . .</td>
<td>16</td>
</tr>
<tr>
<td>3.4</td>
<td>Beyond SG: Noise Reduction and Second-Order Methods . . . . .</td>
<td>19</td>
</tr>
</table>

---

\*Facebook AI Research, New York, NY, USA. E-mail: [leon@bottou.org](mailto:leon@bottou.org)

<sup>†</sup>Department of Industrial and Systems Engineering, Lehigh University, Bethlehem, PA, USA. Supported by U.S. Department of Energy grant [de-sc0010615](#) and U.S. National Science Foundation grant DMS-1016291. E-mail: [frank.e.curtis@gmail.com](mailto:frank.e.curtis@gmail.com)

<sup>‡</sup>Department of Industrial Engineering and Management Sciences, Northwestern University, Evanston, IL, USA. Supported by Office of Naval Research grant N00014-14-1-0313 P00003 and Department of Energy grant DE-FG02-87ER25047. E-mail: [j-nocedal@northwestern.edu](mailto:j-nocedal@northwestern.edu)


---

## Page 2

<table>
<tr>
<td><b>4</b></td>
<td><b>Analyses of Stochastic Gradient Methods</b></td>
<td><b>21</b></td>
</tr>
<tr>
<td>4.1</td>
<td>Two Fundamental Lemmas . . . . .</td>
<td>23</td>
</tr>
<tr>
<td>4.2</td>
<td>SG for Strongly Convex Objectives . . . . .</td>
<td>25</td>
</tr>
<tr>
<td>4.3</td>
<td>SG for General Objectives . . . . .</td>
<td>31</td>
</tr>
<tr>
<td>4.4</td>
<td>Work Complexity for Large-Scale Learning . . . . .</td>
<td>34</td>
</tr>
<tr>
<td>4.5</td>
<td>Commentary . . . . .</td>
<td>37</td>
</tr>
<tr>
<td><b>5</b></td>
<td><b>Noise Reduction Methods</b></td>
<td><b>40</b></td>
</tr>
<tr>
<td>5.1</td>
<td>Reducing Noise at a Geometric Rate . . . . .</td>
<td>41</td>
</tr>
<tr>
<td>5.2</td>
<td>Dynamic Sample Size Methods . . . . .</td>
<td>42</td>
</tr>
<tr>
<td>5.2.1</td>
<td>Practical Implementation . . . . .</td>
<td>44</td>
</tr>
<tr>
<td>5.3</td>
<td>Gradient Aggregation . . . . .</td>
<td>46</td>
</tr>
<tr>
<td>5.3.1</td>
<td>SVRG . . . . .</td>
<td>46</td>
</tr>
<tr>
<td>5.3.2</td>
<td>SAGA . . . . .</td>
<td>47</td>
</tr>
<tr>
<td>5.3.3</td>
<td>Commentary . . . . .</td>
<td>49</td>
</tr>
<tr>
<td>5.4</td>
<td>Iterate Averaging Methods . . . . .</td>
<td>49</td>
</tr>
<tr>
<td><b>6</b></td>
<td><b>Second-Order Methods</b></td>
<td><b>50</b></td>
</tr>
<tr>
<td>6.1</td>
<td>Hessian-Free Inexact Newton Methods . . . . .</td>
<td>52</td>
</tr>
<tr>
<td>6.1.1</td>
<td>Subsampled Hessian-Free Newton Methods . . . . .</td>
<td>53</td>
</tr>
<tr>
<td>6.1.2</td>
<td>Dealing with Nonconvexity . . . . .</td>
<td>55</td>
</tr>
<tr>
<td>6.2</td>
<td>Stochastic Quasi-Newton Methods . . . . .</td>
<td>55</td>
</tr>
<tr>
<td>6.2.1</td>
<td>Deterministic to Stochastic . . . . .</td>
<td>56</td>
</tr>
<tr>
<td>6.2.2</td>
<td>Algorithms . . . . .</td>
<td>57</td>
</tr>
<tr>
<td>6.3</td>
<td>Gauss-Newton Methods . . . . .</td>
<td>59</td>
</tr>
<tr>
<td>6.4</td>
<td>Natural Gradient Method . . . . .</td>
<td>61</td>
</tr>
<tr>
<td>6.5</td>
<td>Methods that Employ Diagonal Scalings . . . . .</td>
<td>64</td>
</tr>
<tr>
<td><b>7</b></td>
<td><b>Other Popular Methods</b></td>
<td><b>68</b></td>
</tr>
<tr>
<td>7.1</td>
<td>Gradient Methods with Momentum . . . . .</td>
<td>68</td>
</tr>
<tr>
<td>7.2</td>
<td>Accelerated Gradient Methods . . . . .</td>
<td>70</td>
</tr>
<tr>
<td>7.3</td>
<td>Coordinate Descent Methods . . . . .</td>
<td>70</td>
</tr>
<tr>
<td><b>8</b></td>
<td><b>Methods for Regularized Models</b></td>
<td><b>74</b></td>
</tr>
<tr>
<td>8.1</td>
<td>First-Order Methods for Generic Convex Regularizers . . . . .</td>
<td>75</td>
</tr>
<tr>
<td>8.1.1</td>
<td>Iterative Soft-Thresholding Algorithm (ISTA) . . . . .</td>
<td>77</td>
</tr>
<tr>
<td>8.1.2</td>
<td>Bound-Constrained Methods for <math>\ell_1</math>-norm Regularized Problems . . . . .</td>
<td>77</td>
</tr>
<tr>
<td>8.2</td>
<td>Second-Order Methods . . . . .</td>
<td>78</td>
</tr>
<tr>
<td>8.2.1</td>
<td>Proximal Newton Methods . . . . .</td>
<td>79</td>
</tr>
<tr>
<td>8.2.2</td>
<td>Orthant-Based Methods . . . . .</td>
<td>80</td>
</tr>
<tr>
<td><b>9</b></td>
<td><b>Summary and Perspectives</b></td>
<td><b>82</b></td>
</tr>
<tr>
<td><b>A</b></td>
<td><b>Convexity and Analyses of SG</b></td>
<td><b>82</b></td>
</tr>
<tr>
<td><b>B</b></td>
<td><b>Proofs</b></td>
<td><b>83</b></td>
</tr>
</table>


---

## Page 3

# 1 Introduction

The promise of artificial intelligence has been a topic of both public and private interest for decades. Starting in the 1950s, there has been great hope that classical artificial intelligence techniques based on logic, knowledge representation, reasoning, and planning would result in revolutionary software that could, amongst other things, understand language, control robots, and provide expert advice. Although advances based on such techniques may be in store in the future, many researchers have started to doubt these classical approaches, choosing instead to focus their efforts on the design of systems based on statistical techniques, such as in the rapidly evolving and expanding field of *machine learning*.

Machine learning and the intelligent systems that have been borne out of it—such as search engines, recommendation platforms, and speech and image recognition software—have become an indispensable part of modern society. Rooted in statistics and relying heavily on the efficiency of numerical algorithms, machine learning techniques capitalize on the world’s increasingly powerful computing platforms and the availability of datasets of immense size. In addition, as the fruits of its efforts have become so easily accessible to the public through various modalities—such as *the cloud*—interest in machine learning is bound to continue its dramatic rise, yielding further societal, economic, and scientific impacts.

One of the pillars of machine learning is *mathematical optimization*, which, in this context, involves the numerical computation of parameters for a system designed to make decisions based on yet unseen data. That is, based on currently available data, these parameters are chosen to be optimal with respect to a given learning problem. The success of certain optimization methods for machine learning has inspired great numbers in various research communities to tackle even more challenging machine learning problems, and to design new methods that are more widely applicable.

The purpose of this paper is to provide a review and commentary on the past, present, and future of the use of numerical optimization algorithms in the context of machine learning applications. A major theme of this work is that large-scale machine learning represents a distinctive setting in which traditional nonlinear optimization techniques typically falter, and so should be considered secondary to alternative classes of approaches that respect the statistical nature of the underlying problem of interest.

Overall, this paper attempts to provide answers for the following questions.

1. 1. How do optimization problems arise in machine learning applications and what makes them challenging?
2. 2. What have been the most successful optimization methods for large-scale machine learning and why?
3. 3. What recent advances have been made in the design of algorithms and what are open questions in this research area?

We answer the first question with the aid of two case studies. The first, a study of text classification, represents an application in which the success of machine learning has been widely recognized and celebrated. The second, a study of perceptual tasks such as speech or image recognition, represents an application in which machine learning still has had great success, but in a much more enigmatic manner that leaves many questions unanswered. These case studies also illustrate the


---

## Page 4

variety of optimization problems that arise in machine learning: the first involves *convex* optimization problems—derived from the use of logistic regression or support vector machines—while the second typically involves *highly nonlinear and nonconvex* problems—derived from the use of deep neural networks.

With these case studies in hand, we turn our attention to the latter two questions on optimization algorithms, the discussions around which represent the bulk of the paper. Whereas traditional gradient-based methods may be effective for solving small-scale learning problems in which a *batch* approach may be used, in the context of large-scale machine learning it has been a *stochastic* algorithm—namely, the *stochastic gradient* method (SG) proposed by Robbins and Monro [130]—that has been the core strategy of interest. Due to this central role played by SG, we discuss its fundamental theoretical and practical properties within a few contexts of interest. We also discuss recent trends in the design of optimization methods for machine learning, organizing them according to their relationship to SG. We discuss: (i) noise reduction methods that attempt to borrow from the strengths of batch methods, such as their fast convergence rates and ability to exploit parallelism; (ii) methods that incorporate approximate second-order derivative information with the goal of dealing with nonlinearity and ill-conditioning; and (iii) methods for solving regularized problems designed to avoid overfitting and allow for the use of high-dimensional models. Rather than contrast SG and other methods based on the results of numerical experiments—which might bias our review toward a limited test set and implementation details—we focus our attention on fundamental computational trade-offs and theoretical properties of optimization methods.

We close the paper with a look back at what has been learned, as well as additional thoughts about the future of optimization methods for machine learning.

## 2 Machine Learning Case Studies

Optimization problems arise throughout machine learning. We provide two case studies that illustrate their role in the selection of prediction functions in state-of-the-art machine learning systems. We focus on cases that involve very large datasets and for which the number of model parameters to be optimized is also large. By remarking on the structure and scale of such problems, we provide a glimpse into the challenges that make them difficult to solve.

### 2.1 Text Classification via Convex Optimization

The assignment of natural language text to predefined classes based on their contents is one of the fundamental tasks of information management [56]. Consider, for example, the task of determining whether a text document is one that discusses politics. Educated humans can make a determination of this type unambiguously, say by observing that the document contains the names of well-known politicians. Early text classification systems attempted to consolidate knowledge from human experts by building on such observations to formally characterize the word sequences that signify a discussion about a topic of interest (e.g., politics). Unfortunately, however, concise characterizations of this type are difficult to formulate. Rules about which word sequences do or do not signify a topic need to be refined when new documents arise that cannot be classified accurately based on previously established rules. The need to coordinate such a growing collection of possibly contradictory rules limits the applicability of such systems to relatively simple tasks.

By contrast, the statistical machine learning approach begins with the collection of a sizeable


---

## Page 5

set of examples  $\{(x_1, y_1), \dots, (x_n, y_n)\}$ , where for each  $i \in \{1, \dots, n\}$  the vector  $x_i$  represents the *features* of a text document (e.g., the words it includes) and the scalar  $y_i$  is a *label* indicating whether the document belongs ( $y_i = 1$ ) or not ( $y_i = -1$ ) to a particular class (i.e., topic of interest). With such a set of examples, one can construct a classification program, defined by a *prediction function*  $h$ , and measure its performance by counting how often the program prediction  $h(x_i)$  differs from the correct prediction  $y_i$ . In this manner, it seems judicious to search for a prediction function that minimizes the frequency of observed misclassifications, otherwise known as the *empirical risk* of misclassification:

$$R_n(h) = \frac{1}{n} \sum_{i=1}^n \mathbb{1}[h(x_i) \neq y_i], \quad \text{where} \quad \mathbb{1}[A] = \begin{cases} 1 & \text{if } A \text{ is true,} \\ 0 & \text{otherwise.} \end{cases} \quad (2.1)$$

The idea of minimizing such a function gives rise to interesting conceptual issues. Consider, for example, a function that simply memorizes the examples, such as

$$h^{\text{rote}}(x) = \begin{cases} y_i & \text{if } x = x_i \text{ for some } i \in \{1, \dots, n\}, \\ \pm 1 & \text{(arbitrarily) otherwise.} \end{cases} \quad (2.2)$$

This prediction function clearly minimizes (2.1), but it offers no performance guarantees on documents that do not appear in the examples. To avoid such rote memorization, one should aim to find a prediction function that *generalizes* the concepts that may be learned from the examples.

One way to achieve good generalized performance is to choose amongst a carefully selected class of prediction functions, perhaps satisfying certain smoothness conditions and enjoying a convenient parametric representation. How such a class of functions may be selected is not straightforward; we discuss this issue in more detail in §2.3. For now, we only mention that common practice for choosing between prediction functions belonging to a given class is to compare them using *cross-validation* procedures that involve splitting the examples into three disjoint subsets: a *training set*, a *validation set*, and a *testing set*. The process of optimizing the choice of  $h$  by minimizing  $R_n$  in (2.1) is carried out on the training set over the set of candidate prediction functions, the goal of which is to pinpoint a small subset of viable candidates. The generalized performance of each of these remaining candidates is then estimated using the validation set, the best performing of which is chosen as the selected function. The testing set is only used to estimate the generalized performance of this selected function.

Such experimental investigations have, for instance, shown that the *bag of words* approach works very well for text classification [56, 81]. In such an approach, a text document is represented by a feature vector  $x \in \mathbb{R}^d$  whose components are associated with a prescribed set of vocabulary words; i.e., each nonzero component indicates that the associated word appears in the document. (The capabilities of such a representation can also be increased by augmenting the set of words with entries representing short sequences of words.) This encoding scheme leads to very sparse vectors. For instance, the canonical encoding for the standard RCV1 dataset [94] uses a vocabulary of  $d = 47,152$  words to represent news stories that typically contain fewer than 1,000 words. Scaling techniques can be used to give more weight to distinctive words, while scaling to ensure that each document has  $\|x\| = 1$  can be performed to compensate for differences in document lengths [94].

Thanks to such a high-dimensional sparse representation of documents, it has been deemed empirically sufficient to consider prediction functions of the form  $h(x; w, \tau) = w^T x - \tau$ . Here,  $w^T x$  is a linear discriminant parameterized by  $w \in \mathbb{R}^d$  and  $\tau \in \mathbb{R}$  is a bias that provides a way to


---

## Page 6

compromise between precision and recall.<sup>1</sup> The accuracy of the predictions could be determined by counting the number of times that  $\text{sign}(h(x; w, \tau))$  matches the correct label, i.e., 1 or  $-1$ . However, while such a prediction function may be appropriate for classifying new documents, formulating an optimization problem around it to choose the parameters  $(w, \tau)$  is impractical in large-scale settings due to the combinatorial structure introduced by the sign function, which is discontinuous. Instead, one typically employs a continuous approximation through a *loss* function that measures a cost for predicting  $h$  when the true label is  $y$ ; e.g., one may choose a *log-loss* function of the form  $\ell(h, y) = \log(1 + \exp(-hy))$ . Moreover, one obtains a class of prediction functions via the addition of a regularization term parameterized by a scalar  $\lambda > 0$ , leading to a convex optimization problem:

$$\min_{(w, \tau) \in \mathbb{R}^d \times \mathbb{R}} \frac{1}{n} \sum_{i=1}^n \ell(h(x_i; w, \tau), y_i) + \frac{\lambda}{2} \|w\|_2^2. \quad (2.3)$$

This problem may be solved multiple times for a given training set with various values of  $\lambda > 0$ , with the ultimate solution  $(w_*, \tau_*)$  being the one that yields the best performance on a validation set.

Many variants of problem (2.3) have also appeared in the literature. For example, the well-known support vector machine problem [38] amounts to using the *hinge loss*  $\ell(h, y) = \max(0, 1 - hy)$ . Another popular feature is to use an  $\ell_1$ -norm regularizer, namely  $\lambda \|w\|_1$ , which favors sparse solutions and therefore restricts attention to a subset of vocabulary words. Other more sophisticated losses can also target a specific precision/recall tradeoff or deal with hierarchies of classes; e.g., see [65, 49]. The choice of loss function is usually guided by experimentation.

In summary, both theoretical arguments [38] and experimental evidence [81] indicate that a carefully selected family of prediction functions and such high-dimensional representations of documents leads to good performance while avoiding *overfitting*—recall (2.2)—of the training set. The use of a simple surrogate, such as (2.3), facilitates experimentation and has been very successful for a variety of problems even beyond text classification. Simple surrogate models are, however, not the most effective in all applications. For example, for certain perceptual tasks, the use of deep neural networks—which lead to large-scale, highly nonlinear, and *nonconvex* optimization problems—has produced great advances that are unmatched by approaches involving simpler, convex models. We discuss such problems next.

## 2.2 Perceptual Tasks via Deep Neural Networks

Just like text classification tasks, perceptual tasks such as speech and image recognition are not well performed in an automated manner using computer programs based on sets of prescribed rules. For instance, the infinite diversity of writing styles easily defeats attempts to concisely specify which pixel combinations represent the digit *four*; see Figure 2.1. One may attempt to design heuristic techniques to handle such eclectic instantiations of the same object, but as previous attempts to design such techniques have often failed, computer vision researchers are increasingly embracing machine learning techniques.

During the past five years, spectacular applicative successes on perceptual problems have been achieved by machine learning techniques through the use of *deep neural networks* (DNNs). Although there are many kinds of DNNs, most recent advances have been made using essentially the same

---

<sup>1</sup>Precision and recall, defined as the probabilities  $\mathbb{P}[y = 1|h(x) = 1]$  and  $\mathbb{P}[h(x) = 1|y = 1]$ , respectively, are convenient measures of classifier performance when the class of interest represents only a small fraction of all documents.


---

## Page 7

![A row of 12 images representing the number 4 in various styles: five different handwritten cursive '4's, a pixelated '4', a stylized '4' with a green plant growing from it, and five different serif '4's.](41e484d4a58a3c086a464d5f3a71719a_1_img.webp)

Fig. 2.1: No known prescribed rules express all pixel combinations that represent *four*.

types that were popular in the 1990s and almost forgotten in the 2000s [111]. What have made recent successes possible are the availability of much larger datasets and greater computational resources.

Because DNNs were initially inspired by simplified models of biological neurons [134, 135], they are often described with jargon borrowed from neuroscience. For the most part, this jargon turns into a rather effective language for describing a prediction function  $h$  whose value is computed by applying successive transformations to a given input vector  $x_i \in \mathbb{R}^{d_0}$ . These transformations are made in *layers*. For example, a canonical fully connected layer performs the computation

$$x_i^{(j)} = s(W_j x_i^{(j-1)} + b_j) \in \mathbb{R}^{d_j}, \quad (2.4)$$

where  $x_i^{(0)} = x_i$ , the matrix  $W_j \in \mathbb{R}^{d_j \times d_{j-1}}$  and vector  $b_j \in \mathbb{R}^{d_j}$  contain the  $j$ th layer parameters, and  $s$  is a component-wise nonlinear *activation* function. Popular choices for the activation function include the sigmoid function  $s(x) = 1/(1 + \exp(-x))$  and the hinge function  $s(x) = \max\{0, x\}$  (often called a rectified linear unit (ReLU) in this context). In this manner, the ultimate output vector  $x_i^{(J)}$  leads to the prediction function value  $h(x_i; w)$ , where the parameter vector  $w$  collects all the parameters  $\{(W_1, b_1), \dots, (W_J, b_J)\}$  of the successive layers.

Similar to (2.3), an optimization problem in this setting involves the collection of a training set  $\{(x_1, y_1) \dots (x_n, y_n)\}$  and the choice of a loss function  $\ell$  leading to

$$\min_{w \in \mathbb{R}^d} \frac{1}{n} \sum_{i=1}^n \ell(h(x_i; w), y_i). \quad (2.5)$$

However, in contrast to (2.3), this optimization problem is highly nonlinear and nonconvex, making it intractable to solve to global optimality. That being said, machine learning experts have made great strides in the use of DNNs by computing approximate solutions by gradient-based methods. This has been made possible by the conceptually straightforward, yet crucial observation that the gradient of the objective in (2.5) with respect to the parameter vector  $w$  can be computed by the chain rule using algorithmic differentiation [71]. This differentiation technique is known in the machine learning community as *back propagation* [134, 135].

The number of layers in a DNN and the size of each layer are usually determined by performing comparative experiments and evaluating the system performance on a validation set, as in the procedure in §2.1. A contemporary fully connected neural network for speech recognition typically has five to seven layers. This amounts to tens of millions of parameters to be optimized, the training of which may require up to thousands of hours of speech data (representing hundreds of millions of training examples) and weeks of computation on a supercomputer. Figure 2.2 illustrates the word error rate gains achieved by using DNNs for acoustic modeling in three state-of-the-art speech recognition systems. These gains in accuracy are so significant that DNNs are now used in all the main commercial speech recognition products.


---

## Page 8

![Bar chart showing Word error rates for GMM and DNN across three benchmarks: Switchboard (Microsoft), Youtube (Google), and Broadcast News (IBM).](7200c1c6e53b8ac02b57df152033edc6_1_img.webp)

<table border="1">
<thead>
<tr>
<th>Benchmark</th>
<th>GMM (%)</th>
<th>DNN (%)</th>
</tr>
</thead>
<tbody>
<tr>
<td>Switchboard (Microsoft)</td>
<td>27.4%</td>
<td>18.5%</td>
</tr>
<tr>
<td>Youtube (Google)</td>
<td>52.3%</td>
<td>47.6%</td>
</tr>
<tr>
<td>Broadcast News (IBM)</td>
<td>17.2%</td>
<td>14.9%</td>
</tr>
</tbody>
</table>

Fig. 2.2: Word error rates reported by three different research groups on three standard speech recognition benchmarks. For all three groups, deep neural networks (DNNs) significantly outperform the traditional Gaussian mixture models (GMMs) [50]. These experiments were performed between 2010 and 2012 and were instrumental in the recent revival of DNNs.

At the same time, convolutional neural networks (CNNs) have proved to be very effective for computer vision and signal processing tasks [87, 24, 88, 85]. Such a network is composed of convolutional layers, wherein the parameter matrix  $W_j$  is a circulant matrix and the input  $x_i^{(j-1)}$  is interpreted as a multichannel image. The product  $W_j x_i^{(j-1)}$  then computes the convolution of the image by a trainable filter while the activation function—which are piecewise linear functions as opposed to sigmoids—can perform more complex operations that may be interpreted as image rectification, contrast normalization, or subsampling. Figure 2.3 represents the architecture of the winner of the landmark 2012 ImageNet Large Scale Visual Recognition Competition (ILSVRC) [148]. The figure illustrates a CNN with five convolutional layers and three fully connected layers [85]. The input vector represents the pixel values of a  $224 \times 224$  image while the output scores represent the odds that the image belongs to each of 1,000 categories. This network contains about 60 million parameters, the training of which on a few million labeled images takes a few days on a dual GPU workstation.

The diagram illustrates the architecture of the 2012 ILSVRC winner CNN, consisting of eight layers. The layers are grouped into two main sections: Convolutional feature extraction layers (C1-C5) and Fully connected layers (F6-F8).

- **Convolutional feature extraction layers (C1-C5):**
  - **Input:** A  $3 \times 224 \times 224$  image, which is  $11 \times 11$  and *strided*.
  - **C1:**  $192 \times 28 \times 28$ , norm+pool. It uses a  $5 \times 5$  filter and a *split* operation.
  - **C2:**  $256 \times 13 \times 13$ , norm+pool. It uses a  $3 \times 3$  filter and a *split* operation.
  - **C3:**  $384 \times 13 \times 13$ . It uses a  $3 \times 3$  filter and a *split* operation.
  - **C4:**  $384 \times 13 \times 13$ . It uses a  $3 \times 3$  filter and a *split* operation.
  - **C5:**  $256 \times 6 \times 6$ , pool.
- **Fully connected layers (F6-F8):**
  - **F6:** 4096 dropout.
  - **F7:** 4096 dropout.
  - **F8:** 1000 classes.

Fig. 2.3: Architecture for image recognition. The 2012 ILSVRC winner consists of eight layers [85]. Each layer performs a linear transformation (specifically, convolutions in layers C1–C5 and matrix multiplication in layers F6–F8) followed by nonlinear transformations (rectification in all layers, contrast normalization in C1–C2, and pooling in C1–C2 and C5). Regularization with dropout noise is used in layers F6–F7.


---

## Page 9

Figure 2.4 illustrates the historical error rates of the winner of the 2012 ILSVRC. In this competition, a classification is deemed successful if the correct category appeared among the top five categories returned by the system. The large performance gain achieved in 2012 was confirmed in the following years, and today CNNs are considered the tool of choice for visual object recognition [129]. They are currently deployed by numerous Internet companies for image search and face recognition.

![Bar chart showing the historical top5 error rate of the annual winner of the ImageNet image classification challenge from 2010 to 2015. The chart compares Traditional methods (yellow bars) and CNNs (dark blue bars). The error rate decreases significantly over time, with a major drop in 2012 when CNNs were introduced.](6e30c1b92705e306e8eb96b362e80033_2_img.webp)

<table border="1">
<thead>
<tr>
<th>Year</th>
<th>Traditional Error Rate (%)</th>
<th>CNN Error Rate (%)</th>
</tr>
</thead>
<tbody>
<tr>
<td>2010</td>
<td>28.0%</td>
<td>-</td>
</tr>
<tr>
<td>2011</td>
<td>26.0%</td>
<td>-</td>
</tr>
<tr>
<td>2012</td>
<td>-</td>
<td>15.0%</td>
</tr>
<tr>
<td>2013</td>
<td>-</td>
<td>12.0%</td>
</tr>
<tr>
<td>2014</td>
<td>-</td>
<td>7.0%</td>
</tr>
<tr>
<td>2015</td>
<td>-</td>
<td>4.0%</td>
</tr>
</tbody>
</table>

Fig. 2.4: Historical *top5* error rate of the annual winner of the ImageNet image classification challenge. A convolutional neural network (CNN) achieved a significant performance improvement over all traditional methods in 2012. The following years have cemented CNNs as the current state-of-the-art in visual object recognition [85, 129].

The successes of DNNs in modern machine learning applications are undeniable. Although the training process requires extreme skill and care—e.g., it is crucial to initialize the optimization process with a good starting point and to monitor its progress while correcting conditioning issues as they appear [89]—the mere fact that one can do anything useful with such large, highly nonlinear and nonconvex models is remarkable.

## 2.3 Formal Machine Learning Procedure

Through our case studies, we have illustrated how a process of machine learning leads to the selection of a prediction function  $h$  through solving an optimization problem. Moving forward, it is necessary to formalize our presentation by discussing in greater detail the principles behind the selection process, stressing the theoretical importance of *uniform laws of large numbers* as well as the practical importance of *structural risk minimization*.

For simplicity, we continue to focus on the problems that arise in the context of *supervised classification*; i.e., we focus on the optimization of prediction functions for labeling unseen data based on information contained in a set of labeled training data. Such a focus is reasonable as many unsupervised and other learning techniques reduce to optimization problems of comparable form; see, e.g., [155].

**Fundamentals** Our goal is to determine a prediction function  $h : \mathcal{X} \rightarrow \mathcal{Y}$  from an input space  $\mathcal{X}$  to an output space  $\mathcal{Y}$  such that, given  $x \in \mathcal{X}$ , the value  $h(x)$  offers an accurate prediction about the true output  $y$ . That is, our goal is to choose a prediction function that avoids rote memorization


---

## Page 10

and instead generalizes the concepts that can be learned from a given set of examples. To do this, one should choose the prediction function  $h$  by attempting to minimize a risk measure over an adequately selected family of prediction functions [158], call it  $\mathcal{H}$ .

To formalize this idea, suppose that the examples are sampled from a joint probability distribution function  $P(x, y)$  that simultaneously represents the distribution  $P(x)$  of inputs as well as the conditional probability  $P(y|x)$  of the label  $y$  being appropriate for an input  $x$ . (With this view, one often refers to the examples as *samples*; we use both terms throughout the rest of the paper.) Rather than one that merely minimizes the empirical risk (2.1), one should seek to find  $h$  that yields a small *expected risk* of misclassification *over all possible inputs*, i.e., an  $h$  that minimizes

$$R(h) = \mathbb{P}[h(x) \neq y] = \mathbb{E}[\mathbb{1}[h(x) \neq y]], \quad (2.6)$$

where  $\mathbb{P}[A]$  and  $\mathbb{E}[A]$  respectively denote the probability and expected value of  $A$ . Such a framework is *variational* since we are optimizing over a set of functions, and is *stochastic* since the objective function involves an expectation.

While one may desire to minimize the expected risk (2.6), in practice one must attempt to do so without explicit knowledge of  $P$ . Instead, the only tractable option is to construct a surrogate problem that relies solely on the examples  $\{(x_i, y_i)\}_{i=1}^n$ . Overall, there are two main issues that must be addressed: (i) how to choose the parameterized family of prediction functions  $\mathcal{H}$  and (ii) how to determine (and find) the particular prediction function  $h \in \mathcal{H}$  that is optimal.

**Choice of Prediction Function Family** The family of functions  $\mathcal{H}$  should be determined with three *potentially competing* goals in mind. First,  $\mathcal{H}$  should contain prediction functions that are able to achieve a low empirical risk over the training set, so as to avoid bias or underfitting the data. This can be achieved by selecting a rich family of functions or by using *a priori* knowledge to select a well-targeted family. Second, the gap between expected risk and empirical risk, namely,  $R(h) - R_n(h)$ , should be small over all  $h \in \mathcal{H}$ . Generally, this gap decreases when one uses more training examples, but, due to potential overfitting, it increases when one uses richer families of functions (see below). This latter fact puts the second goal at odds with the first. Third,  $\mathcal{H}$  should be selected so that one can efficiently solve the corresponding optimization problem, the difficulty of which may increase when one employs a richer family of functions and/or a larger training set.

Our observation about the gap between expected and empirical risk can be understood by recalling certain *laws of large numbers*. For instance, when the expected risk represents a misclassification probability as in (2.6), the Hoeffding inequality [75] guarantees that, with probability at least  $1 - \eta$ , one has

$$|R(h) - R_n(h)| \leq \sqrt{\frac{1}{2n} \log \left( \frac{2}{\eta} \right)} \quad \text{for a given } h \in \mathcal{H}.$$

This bound offers the intuitive explanation that the gap decreases as one uses more training examples. However, this view is insufficient for our purposes since, in the context of machine learning,  $h$  is not a fixed function! Rather,  $h$  is the variable over which one is optimizing.

For this reason, one often turns to *uniform laws of large numbers* and the concept of the Vapnik-Chervonenkis (VC) dimension of  $\mathcal{H}$ , a measure of the *capacity* of such a family of functions [158]. For the intuition behind this concept, consider, e.g., a binary classification scheme in  $\mathbb{R}^2$  where one assigns a label of 1 for points above a polynomial and  $-1$  for points below. The set of linear


---

## Page 11

polynomials has a low capacity in the sense that it is only capable of accurately classifying training points that can be separated by a line; e.g., in two variables, a linear classifier has a VC dimension of three. A set of high-degree polynomials, on the other hand, has a high capacity since it can accurately separate training points that are interspersed; the VC dimension of a polynomial of degree  $D$  in  $d$  variables is  $\binom{d+D}{d}$ . That being said, the gap between empirical and expected risk can be larger for a set of high-degree polynomials since the high capacity allows them to overfit a given set of training data.

Mathematically, with the VC dimension measuring capacity, one can establish one of the most important results in learning theory: with  $d_{\mathcal{H}}$  defined as the VC dimension of  $\mathcal{H}$ , one has with probability at least  $1 - \eta$  that

$$\sup_{h \in \mathcal{H}} |R(h) - R_n(h)| \leq \mathcal{O} \left( \sqrt{\frac{1}{2n} \log \left( \frac{2}{\eta} \right) + \frac{d_{\mathcal{H}}}{n} \log \left( \frac{n}{d_{\mathcal{H}}} \right)} \right). \quad (2.7)$$

This bound gives a more accurate picture of the dependence of the gap on the choice of  $\mathcal{H}$ . For example, it shows that for a fixed  $d_{\mathcal{H}}$ , uniform convergence is obtained by increasing the number of training points  $n$ . However, it also shows that, for a fixed  $n$ , the gap can widen for larger  $d_{\mathcal{H}}$ . Indeed, to maintain the same gap, one must increase  $n$  at the same rate if  $d_{\mathcal{H}}$  is increased. The uniform convergence embodied in this result is crucial in machine learning since one wants to ensure that the prediction system performs well with any data provided to it. In §4.4, we employ a slight variant of this result to discuss computational trade-offs that arise in large-scale learning.<sup>2</sup>

Interestingly, one quantity that does not enter in (2.7) is the number of parameters that distinguish a particular member function  $h$  of the family  $\mathcal{H}$ . In some settings such as logistic regression, this number is essentially the same as  $d_{\mathcal{H}}$ , which might suggest that the task of optimizing over  $h \in \mathcal{H}$  is more cumbersome as  $d_{\mathcal{H}}$  increases. However, this is not always the case. Certain families of functions are amenable to minimization despite having a very large or even infinite number of parameters [156, Section 4.11]. For example, support vector machines [38] were designed to take advantage of this fact [156, Theorem 10.3].

All in all, while bounds such as (2.7) are theoretically interesting and provide useful insight, they are rarely used directly in practice since, as we have suggested in §2.1 and §2.2, it is typically easier to estimate the gap between empirical and expected risk with *cross-validation* experiments. We now present ideas underlying a practical framework that respects the trade-offs mentioned above.

**Structural Risk Minimization** An approach for choosing a prediction function that has proved to be widely successful in practice is *structural risk minimization* [157, 156]. Rather than choose a generic family of prediction functions—over which it would be both difficult to optimize and to estimate the gap between empirical and expected risks—one chooses a *structure*, i.e., a collection of nested function families. For instance, such a structure can be formed as a collection of subsets of a given family  $\mathcal{H}$  in the following manner: given a preference function  $\Omega$ , choose various values of a *hyperparameter*  $C$ , according to each of which one obtains the subset  $\mathcal{H}_C := \{h \in \mathcal{H} : \Omega(h) \leq C\}$ . Given a fixed number of examples, increasing  $C$  reduces the empirical risk (i.e., the minimum of

---

<sup>2</sup>We also note that considerably better bounds hold when one can collect statistics on actual examples, e.g., by determining gaps dependent on an observed variance of the risk or by considering uniform bounds restricted to families of prediction functions that achieve a risk within a certain threshold of the optimum [55, 102, 26].


---

## Page 12

$R_n(h)$  over  $h \in \mathcal{H}_C$ ), but, after some point, it typically increases the gap between expected and empirical risks. This phenomenon is illustrated in Figure 2.5.

Other ways to introduce structures are to consider a regularized empirical risk  $R_n(h) + \lambda\Omega(h)$  (an idea introduced in problem (2.3), which may be viewed as the Lagrangian for minimizing  $R_n(h)$  subject to  $\Omega(h) \leq C$ ), enlarge the dictionary in a bag-of-words representation, increase the degree of a polynomial model function, or add to the dimension of an inner layer of a DNN.

![Figure 2.5: Illustration of structural risk minimization. The graph plots Misclassification rate on the y-axis against C on the x-axis. Two curves are shown: a solid red line labeled 'Observed empirical risk R_n(w)' and a dashed blue line labeled 'Guaranteed expected risk R(w)'. The red curve decreases as C increases. The blue curve starts high, decreases to a minimum, and then increases. A light blue callout box points to the minimum of the blue curve, stating: 'This value of C here gives the best guarantee for the expected risk R(w).'](2d23fb9e24116f506c744ea171465f60_3_img.webp)

Fig. 2.5: Illustration of structural risk minimization. Given a set of  $n$  examples, a decision function family  $\mathcal{H}$ , and a relative preference function  $\Omega$ , the figure illustrates a typical relationship between the expected and empirical risks corresponding to a prediction function obtained by an optimization algorithm that minimizes an empirical risk  $R_n(h)$  subject to  $\Omega(h) \leq C$ . The optimal empirical risk decreases when  $C$  increases. Meanwhile, the deviation between empirical and expected risk is bounded above by a quantity—which depends on  $\mathcal{H}$  and  $\Omega$ —that increases with  $C$ . While not shown in the figure, the value of  $C$  that offers the best guarantee on the expected risk increases with  $n$ , i.e., the number of examples; recall (2.7).

Given such a set-up, one can avoid estimating the gap between empirical and expected risk by splitting the available data into subsets: a *training set* used to produce a subset of candidate solutions, a *validation set* used to estimate the expected risk for each such candidate, and a *testing set* used to estimate the expected risk for the candidate that is ultimately chosen. Specifically, over the training set, one minimizes an empirical risk measure  $R_n$  over  $\mathcal{H}_C$  for various values of  $C$ . This results in a handful of candidate functions. The validation set is then used to estimate the expected risk corresponding to each candidate solution, after which one chooses the function yielding the lowest estimated risk value. Assuming a large enough range for  $C$  has been used, one often finds that the best solution does not correspond to the largest value of  $C$  considered; again, see Figure 2.5.

Another, albeit indirect avenue toward risk minimization is to employ an algorithm for minimizing  $R_n$ , but terminate the algorithm *early*, i.e., before an actual minimizer of  $R_n$  is found. In this manner, the role of the hyperparameter is played by the training time allowed, according to which one typically finds the relationships illustrated in Figure 2.6. Theoretical analyses related to the idea of early stopping are much more challenging than those for other forms of structural risk minimization. However, it is worthwhile to mention these effects since early stopping is a popular technique in practice, and is often *essential* due to computational budget limitations.

Overall, the structural risk minimization principle has proved useful for many applications, and


---

## Page 13

![Figure 2.6: Illustration of early stopping. A graph showing Risk on the y-axis and Training time on the x-axis. Two curves are plotted: a solid red line for 'Observed empirical risk R_n(w)' and a dashed blue line for 'Observed expected risk R(w)'. Both curves start high and decrease as training time increases. The red curve decreases more slowly than the blue curve. A callout box points to the minimum of the blue curve, stating: 'Stopping the optimization here achieves better expected risk R(w).](efbc20d54002ab4c19fb4d430e5a9adb_1_img.webp)

Fig. 2.6: Illustration of early stopping. Prematurely stopping the optimization of the empirical risk  $R_n$  often results in a better expected risk  $R$ . In this manner, the stopping time plays a similar role as the hyperparameter  $C$  in the illustration of structural risk minimization in Figure 2.5.

can be viewed as an alternative of the approach of employing expert human knowledge mentioned in §2.1. Rather than encoding knowledge as formal classification rules, one encodes it via preferences for certain prediction functions over others, then explores the performance of various prediction functions that have been optimized under the influence of such preferences.

### 3 Overview of Optimization Methods

We now turn our attention to the main focus of our study, namely, numerical algorithms for solving optimization problems that arise in large-scale machine learning. We begin by formalizing our problems of interest, which can be seen as generic statements of problems of the type described in §2 for minimizing expected and empirical risks. We then provide an overview of two main classes of optimization methods—*stochastic* and *batch*—that can be applied to solve such problems, emphasizing some of the fundamental reasons why stochastic methods have inherent advantages. We close this section with a preview of some of the advanced optimization techniques that are discussed in detail in later sections, which borrow ideas from both stochastic and batch methods.

#### 3.1 Formal Optimization Problem Statements

As seen in §2, optimization problems in machine learning arise through the definition of prediction and loss functions that appear in measures of expected and empirical risk that one aims to minimize. Our discussions revolve around the following definitions.

**Prediction and Loss Functions** Rather than consider a variational optimization problem over a generic family of prediction functions, we assume that the prediction function  $h$  has a fixed form and is parameterized by a real vector  $w \in \mathbb{R}^d$  over which the optimization is to be performed. Formally, for some given  $h(\cdot; \cdot) : \mathbb{R}^{d_x} \times \mathbb{R}^d \rightarrow \mathbb{R}^{d_y}$ , we consider the family of prediction functions

$$\mathcal{H} := \{h(\cdot; w) : w \in \mathbb{R}^d\}.$$


---

## Page 14

We aim to find the prediction function in this family that minimizes the losses incurred from inaccurate predictions. For this purpose, we assume a given loss function  $\ell : \mathbb{R}^{d_y} \times \mathbb{R}^{d_y} \rightarrow \mathbb{R}$  as one that, given an input-output pair  $(x, y)$ , yields the loss  $\ell(h(x; w), y)$  when  $h(x; w)$  and  $y$  are the predicted and true outputs, respectively.

**Expected Risk** Ideally, the parameter vector  $w$  is chosen to minimize the expected loss that would be incurred from *any* input-output pair. To state this idea formally, we assume that losses are measured with respect to a probability distribution  $P(x, y)$  representing the true relationship between inputs and outputs. That is, we assume that the input-output space  $\mathbb{R}^{d_x} \times \mathbb{R}^{d_y}$  is endowed with  $P : \mathbb{R}^{d_x} \times \mathbb{R}^{d_y} \rightarrow [0, 1]$  and the objective function we wish to minimize is

$$R(w) = \int_{\mathbb{R}^{d_x} \times \mathbb{R}^{d_y}} \ell(h(x; w), y) dP(x, y) = \mathbb{E}[\ell(h(x; w), y)]. \quad (3.1)$$

We say that  $R : \mathbb{R}^d \rightarrow \mathbb{R}$  yields the *expected risk* (i.e., expected loss) given a parameter vector  $w$  with respect to the probability distribution  $P$ .

**Empirical Risk** While it may be desirable to minimize (3.1), such a goal is untenable when one does not have complete information about  $P$ . Thus, in practice, one seeks the solution of a problem that involves an estimate of the expected risk  $R$ . In supervised learning, one has access (either all-at-once or incrementally) to a set of  $n \in \mathbb{N}$  independently drawn input-output samples  $\{(x_i, y_i)\}_{i=1}^n \subseteq \mathbb{R}^{d_x} \times \mathbb{R}^{d_y}$ , with which one may define the *empirical risk* function  $R_n : \mathbb{R}^d \rightarrow \mathbb{R}$  by

$$R_n(w) = \frac{1}{n} \sum_{i=1}^n \ell(h(x_i; w), y_i). \quad (3.2)$$

Generally speaking, minimization of  $R_n$  may be considered the practical optimization problem of interest. For now, we consider the unregularized measure (3.2), remarking that the optimization methods that we discuss in the subsequent sections can be applied readily when a smooth regularization term is included. (We leave a discussion of nonsmooth regularizers until §8.)

Note that, in §2, the functions  $R$  and  $R_n$  represented *misclassification error*; see (2.1) and (2.6). However, these new definitions of  $R$  and  $R_n$  measure the loss as determined by the function  $\ell$ . We use these latter definitions for the rest of the paper.

**Simplified Notation** The expressions (3.1) and (3.2) show explicitly how the expected and empirical risks depend on the loss function, sample space or sample set, etc. However, when discussing optimization methods, we will often employ a simplified notation that also offers some avenues for generalizing certain algorithmic ideas. In particular, let us represent a sample (or set of samples) by a random seed  $\xi$ ; e.g., one may imagine a realization of  $\xi$  as a single sample  $(x, y)$  from  $\mathbb{R}^{d_x} \times \mathbb{R}^{d_y}$ , or a realization of  $\xi$  might be a set of samples  $\{(x_i, y_i)\}_{i \in \mathcal{S}}$ . In addition, let us refer to the loss incurred for a given  $(w, \xi)$  as  $f(w; \xi)$ , i.e.,

$$f \text{ is the composition of the loss function } \ell \text{ and the prediction function } h. \quad (3.3)$$

In this manner, the expected risk for a given  $w$  is the expected value of this composite function taken with respect to the distribution of  $\xi$ :

$$\text{(Expected Risk)} \quad R(w) = \mathbb{E}[f(w; \xi)]. \quad (3.4)$$


---

## Page 15

In a similar manner, when given a set of realizations  $\{\xi_{[i]}\}_{i=1}^n$  of  $\xi$  corresponding to a sample set  $\{(x_i, y_i)\}_{i=1}^n$ , let us define the loss incurred by the parameter vector  $w$  with respect to the  $i$ th sample as

$$f_i(w) := f(w; \xi_{[i]}), \quad (3.5)$$

and then write the empirical risk as the average of the sample losses:

$$\text{(Empirical Risk)} \quad R_n(w) = \frac{1}{n} \sum_{i=1}^n f_i(w). \quad (3.6)$$

For future reference, we use  $\xi_{[i]}$  to denote the  $i$ th element of a fixed set of realizations of a random variable  $\xi$ , whereas, starting in §4, we will use  $\xi_k$  to denote the  $k$ th element of a sequence of random variables.

### 3.2 Stochastic vs. Batch Optimization Methods

Let us now introduce some fundamental optimization algorithms for minimizing risk. For the moment, since it is the typical setting in practice, we introduce two algorithm classes in the context of minimizing the empirical risk measure  $R_n$  in (3.6). Note, however, that much of our later discussion will focus on the performance of algorithms when considering the true measure of interest, namely, the expected risk  $R$  in (3.4).

Optimization methods for machine learning fall into two broad categories. We refer to them as *stochastic* and *batch*. The prototypical stochastic optimization method is the stochastic gradient method (SG) [130], which, in the context of minimizing  $R_n$  and with  $w_1 \in \mathbb{R}^d$  given, is defined by

$$w_{k+1} \leftarrow w_k - \alpha_k \nabla f_{i_k}(w_k). \quad (3.7)$$

Here, for all  $k \in \mathbb{N} := \{1, 2, \dots\}$ , the index  $i_k$  (corresponding to the seed  $\xi_{[i_k]}$ , i.e., the sample pair  $(x_{i_k}, y_{i_k})$ ) is chosen *randomly* from  $\{1, \dots, n\}$  and  $\alpha_k$  is a positive stepsize. Each iteration of this method is thus very cheap, involving only the computation of the gradient  $\nabla f_{i_k}(w_k)$  corresponding to one sample. The method is notable in that the iterate sequence is not determined uniquely by the function  $R_n$ , the starting point  $w_1$ , and the sequence of stepsizes  $\{\alpha_k\}$ , as it would in a deterministic optimization algorithm. Rather,  $\{w_k\}$  is a stochastic process whose behavior is determined by the random sequence  $\{i_k\}$ . Still, as we shall see in our analysis in §4, while each direction  $-\nabla f_{i_k}(w_k)$  might not be one of descent from  $w_k$  (in the sense of yielding a negative directional derivative for  $R_n$  from  $w_k$ ), if it is a descent direction *in expectation*, then the sequence  $\{w_k\}$  can be guided toward a minimizer of  $R_n$ .

For many in the optimization research community, a *batch* approach is a more natural and well-known idea. The simplest such method in this class is the steepest descent algorithm—also referred to as the gradient, batch gradient, or full gradient method—which is defined by the iteration

$$w_{k+1} \leftarrow w_k - \alpha_k \nabla R_n(w_k) = w_k - \frac{\alpha_k}{n} \sum_{i=1}^n \nabla f_i(w_k). \quad (3.8)$$

Computing the step  $-\alpha_k \nabla R_n(w_k)$  in such an approach is more expensive than computing the step  $-\alpha_k \nabla f_{i_k}(w_k)$  in SG, though one may expect that a better step is computed when all samples are considered in an iteration.


---

## Page 16

Stochastic and batch approaches offer different trade-offs in terms of per-iteration costs and expected per-iteration improvement in minimizing empirical risk. Why, then, has SG risen to such prominence in the context of large-scale machine learning? Understanding the reasoning behind this requires careful consideration of the computational trade-offs between stochastic and batch methods, as well as a deeper look into their abilities to guarantee improvement in the underlying expected risk  $R$ . We start to investigate these topics in the next subsection.

We remark in passing that the stochastic and batch approaches mentioned here have analogues in the simulation and stochastic optimization communities, where they are referred to as *stochastic approximation* (SA) and *sample average approximation* (SAA), respectively [63].

#### Inset 3.1: Herbert Robbins and Stochastic Approximation

The paper by Robbins and Monro [130] represents a landmark in the history of numerical optimization methods. Together with the invention of back propagation [134, 135], it also represents one of the most notable developments in the field of machine learning. The SG method was first proposed in [130], not as a gradient method, but as a Markov chain.

Viewed more broadly, the works by Robbins and Monro [130] and Kalman [83] mark the beginning of the field of *stochastic approximation*, which studies the behavior of iterative methods that use noisy signals. The initial focus on optimization led to the study of algorithms that track the solution of the ordinary differential equation  $\dot{w} = -\nabla F(w)$ . Stochastic approximation theory has had a major impact in signal processing and in areas closer to the subject of this paper, such as pattern recognition [4] and neural networks [20].

After receiving his PhD, Herbert Robbins became a lecturer at New York University, where he co-authored with Richard Courant the popular book *What is Mathematics?* [39], which is still in print after more than seven decades [40]. Robbins went on to become one of the most prominent mathematicians of the second half of the twentieth century, known for his contributions to probability, algebra, and graph theory.

### 3.3 Motivation for Stochastic Methods

Before discussing the strengths of stochastic methods such as SG, one should not lose sight of the fact that batch approaches possess some intrinsic advantages. First, when one has reduced the stochastic problem of minimizing the expected risk  $R$  to focus exclusively on the deterministic problem of minimizing the empirical risk  $R_n$ , the use of full gradient information at each iterate opens the door for many deterministic gradient-based optimization methods. That is, in a batch approach, one has at their disposal the wealth of nonlinear optimization techniques that have been developed over the past decades, including the full gradient method (3.8), but also accelerated gradient, conjugate gradient, quasi-Newton, and inexact Newton methods [114]. (See §6 and §7 for discussion of these techniques.) Second, due to the sum structure of  $R_n$ , a batch method can easily benefit from parallelization since the bulk of the computation lies in evaluations of  $R_n$  and  $\nabla R_n$ . Calculations of these quantities can even be done in a distributed manner.

Despite these advantages, there are intuitive, practical, and theoretical reasons for following a stochastic approach. Let us motivate them by contrasting the hallmark SG iteration (3.7) with the full batch gradient iteration (3.8).


---

## Page 17

**Intuitive Motivation** On an intuitive level, SG employs information more efficiently than a batch method. To see this, consider a situation in which a training set, call it  $\mathcal{S}$ , consists of ten copies of a set  $\mathcal{S}_{\text{sub}}$ . A minimizer of empirical risk for the larger set  $\mathcal{S}$  is clearly given by a minimizer for the smaller set  $\mathcal{S}_{\text{sub}}$ , but if one were to apply a batch approach to minimize  $R_n$  over  $\mathcal{S}$ , then each iteration would be *ten times* more expensive than if one only had one copy of  $\mathcal{S}_{\text{sub}}$ . On the other hand, SG performs the same computations in both scenarios, in the sense that the stochastic gradient computations involve choosing elements from  $\mathcal{S}_{\text{sub}}$  with the same probabilities. In reality, a training set typically does not consist of exact duplicates of sample data, but in many large-scale applications the data does involve a good deal of (approximate) redundancy. This suggests that using all of the sample data in every optimization iteration is inefficient.

A similar conclusion can be drawn by recalling the discussion in §2 related to the use of training, validation, and testing sets. If one believes that working with only, say, half of the data in the training set is sufficient to make good predictions on unseen data, then one may argue against working with the entire training set in every optimization iteration. Repeating this argument, working with only a quarter of the training set may be useful at the start, or even with only an eighth of the data, and so on. In this manner, we arrive at motivation for the idea that working with small samples, at least initially, can be quite appealing.

**Practical Motivation** The intuitive benefits just described have been observed repeatedly in practice, where one often finds very real advantages of SG in many applications. As an example, Figure 3.1 compares the performance of a batch L-BFGS method [97, 113] (see §6) and the SG method (3.7) with a constant stepsize (i.e.,  $\alpha_k = \alpha$  for all  $k \in \mathbb{N}$ ) on a binary classification problem using a logistic loss objective function and the data from the RCV1 dataset mentioned in §2.1. The figure plots the empirical risk  $R_n$  as a function of the number of accesses of a sample from the training set, i.e., the number of evaluations of a sample gradient  $\nabla f_{i_k}(w_k)$ . Each set of  $n$  consecutive accesses is called an *epoch*. The batch method performs only one step per epoch while SG performs  $n$  steps per epoch. The plot shows the behavior over the first 10 epochs. The advantage of SG is striking and representative of typical behavior in practice. (One should note, however, that to obtain such efficient behavior, it was necessary to run SG repeatedly using different choices for the stepsize  $\alpha$  until a good choice was identified for this particular problem. We discuss theoretical and practical issues related to the choice of stepsize in our analysis in §4.)

At this point, it is worthwhile to mention that the fast initial improvement achieved by SG, followed by a drastic slowdown after 1 or 2 epochs, is common in practice and is fairly well understood. An intuitive way to explain this behavior is by considering the following example due to Bertsekas [15].

**Example 3.1.** Suppose that each  $f_i$  in (3.6) is a convex quadratic with minimal value at zero and minimizers  $w_{i,*}$  evenly distributed in  $[-1, 1]$  such that the minimizer of  $R_n$  is  $w_* = 0$ ; see Figure 3.2. At  $w_1 \ll -1$ , SG will, with certainty, move to the right (toward  $w_*$ ). Indeed, even if a subsequent iterate lies slightly to the right of the minimizer  $w_{1,*}$  of the “leftmost” quadratic, it is likely (but not certain) that SG will continue moving to the right. However, as iterates near  $w_*$ , the algorithm enters a region of confusion in which there is a significant chance that a step will not move toward  $w_*$ . In this manner, progress will slow significantly. Only with more complete gradient information could the method know with certainty how to move toward  $w_*$ .

Despite the issues illustrated by this example, we shall see in §4 that one can nevertheless ensure


---

## Page 18

![Figure 3.1: A line graph showing Empirical Risk (Y-axis, 0 to 0.6) versus Accessed Data Points (X-axis, 0 to 4 x 10^5). Two curves are plotted: a red curve labeled 'SGD' and a blue curve labeled 'LBFGS'. The SGD curve starts at a risk of approximately 0.65 and drops very sharply to near zero within the first 0.5 x 10^5 data points. The LBFGS curve starts at the same risk of 0.65 but decreases much more gradually, reaching near zero risk only after approximately 2.5 x 10^5 data points.](32e12c3c6e64f7d9f6de9e1e120b02a9_1_img.webp)

Fig. 3.1: Empirical risk  $R_n$  as a function of the number of accessed data points (ADP) for a batch L-BFGS method and the stochastic gradient (SG) method (3.7) on a binary classification problem with a logistic loss objective and the RCV1 dataset. SG was run with a fixed stepsize of  $\alpha = 4$ .

![Figure 3.2: A diagram illustrating the fast initial behavior of the SG method. It shows a horizontal axis with points w_1, -1, w_{1,*}, and 1. Above the axis, there are several overlapping convex quadratic curves, each representing a function f_i. The minimum of one of these curves is at w_{1,*}, which is between -1 and 1.](32e12c3c6e64f7d9f6de9e1e120b02a9_3_img.webp)

Fig. 3.2: Simple illustration to motivate the fast initial behavior of the SG method for minimizing empirical risk (3.6), where each  $f_i$  is a convex quadratic. This example is adapted from [15].

convergence by employing a sequence of diminishing stepsizes to overcome any oscillatory behavior of the algorithm.

**Theoretical Motivation** One can also cite theoretical arguments for a preference of SG over a batch approach. Let us give a preview of these arguments now, which are studied in more depth and further detail in §4.

- • It is well known that a batch approach can minimize  $R_n$  at a fast rate; e.g., if  $R_n$  is strongly convex (see Assumption 4.5) and one applies a batch gradient method, then there exists a constant  $\rho \in (0, 1)$  such that, for all  $k \in \mathbb{N}$ , the *training error* satisfies

$$R_n(w_k) - R_n^* \leq \mathcal{O}(\rho^k), \quad (3.9)$$

where  $R_n^*$  denotes the minimal value of  $R_n$ . The rate of convergence exhibited here is referred to as R-linear convergence in the optimization literature [117] and geometric convergence in the machine learning research community; we shall simply refer to it as *linear convergence*. From (3.9), one can conclude that, in the worst case, the total number of iterations in which the training error can be above a given  $\epsilon > 0$  is proportional to  $\log(1/\epsilon)$ . This means that, with


---

## Page 19

a per-iteration cost proportional to  $n$  (due to the need to compute  $\nabla R_n(w_k)$  for all  $k \in \mathbb{N}$ ), the total work required to obtain  $\epsilon$ -optimality for a batch gradient method is proportional to  $n \log(1/\epsilon)$ .

- • The rate of convergence of a basic stochastic method is slower than for a batch gradient method; e.g., if  $R_n$  is strictly convex and each  $i_k$  is drawn uniformly from  $\{1, \dots, n\}$ , then, for all  $k \in \mathbb{N}$ , the SG iterates defined by (3.7) satisfy the *sublinear convergence* property (see Theorem 4.7)

$$\mathbb{E}[R_n(w_k) - R_n^*] = \mathcal{O}(1/k). \quad (3.10)$$

However, it is crucial to note that *neither the per-iteration cost nor the right-hand side of (3.10) depends on the sample set size  $n$* . This means that the total work required to obtain  $\epsilon$ -optimality for SG is proportional to  $1/\epsilon$ . Admittedly, this can be larger than  $n \log(1/\epsilon)$  for moderate values of  $n$  and  $\epsilon$ , but, as discussed in detail in §4.4, the comparison favors SG when one moves to the *big data* regime where  $n$  is large and one is merely limited by a computational time budget.

- • Another important feature of SG is that, in a stochastic optimization setting, it yields the same convergence rate as in (3.10) for the error in expected risk,  $R - R^*$ , where  $R^*$  is the minimal value of  $R$ . Specifically, by applying the SG iteration (3.7), but with  $\nabla f_{i_k}(w_k)$  replaced by  $\nabla f(w_k; \xi_k)$  with each  $\xi_k$  drawn independently according to the distribution  $P$ , one finds that

$$\mathbb{E}[R(w_k) - R^*] = \mathcal{O}(1/k); \quad (3.11)$$

again a sublinear rate, but on the expected risk. Moreover, in this context, a batch approach is not even viable without the ability to compute  $\nabla R$ . Of course, this represents a different setting than one in which only a finite training set is available, but it reveals that if  $n$  is large with respect to  $k$ , then the behavior of SG in terms of minimizing the empirical risk  $R_n$  or the expected risk  $R$  is practically indistinguishable up to iteration  $k$ . This property cannot be claimed by a batch method.

In summary, there are intuitive, practical, and theoretical arguments in favor of stochastic over batch approaches in optimization methods for large-scale machine learning. For these reasons, and since SG is used so pervasively by practitioners, we frame our discussions about optimization methods in the context of their relationship with SG. We do not claim, however, that batch methods have no place in practice. For one thing, if Figure 3.1 were to consider a larger number of epochs, then one would see the batch approach eventually overtake the stochastic method and yield a lower training error. This motivates why many recently proposed methods try to combine the best properties of batch and stochastic algorithms. Moreover, the SG iteration is difficult to parallelize and requires excessive communication between nodes in a distributed computing setting, providing further impetus for the design of new and improved optimization algorithms.

### 3.4 Beyond SG: Noise Reduction and Second-Order Methods

Looking forward, one of the main questions being asked by researchers and practitioners alike is: what lies beyond SG that can serve as an efficient, reliable, and easy-to-use optimization method for the kinds of applications discussed in §2?


---

## Page 20

To answer this question, we depict in Figure 3.3 methods that aim to improve upon SG as lying on a two-dimensional plane. At the origin of this organizational scheme is SG, representing the base from which all other methods may be compared.

Fig. 3.3: Schematic of a two-dimensional spectrum of optimization methods for machine learning. The horizontal axis represents methods designed to control stochastic noise; the second axis, methods that deal with ill conditioning.

From the origin along the horizontal access, we place methods that are neither purely stochastic nor purely batch, but attempt to combine the best properties of both approaches. For example, observing the iteration (3.7), one quickly realizes that there is no particular reason to employ information from only one sample point per iteration. Instead, one can employ a *mini-batch* approach in which a small subset of samples, call it  $\mathcal{S}_k \subseteq \{1, \dots, n\}$ , is chosen randomly in each iteration, leading to

$$w_{k+1} \leftarrow w_k - \frac{\alpha_k}{|\mathcal{S}_k|} \sum_{i \in \mathcal{S}_k} \nabla f_i(w_k). \quad (3.12)$$

Such an approach falls under the framework set out by Robbins and Monro [130], and allows some degree of parallelization to be exploited in the computation of mini-batch gradients. In addition, one often finds that, due to the reduced variance of the stochastic gradient estimates, the method is easier to tune in terms of choosing the stepsizes  $\{\alpha_k\}$ . Such a mini-batch SG method has been widely used in practice.

Along this horizontal axis, one finds other methods as well. In our investigation, we classify two main groups as *dynamic sample size* and *gradient aggregation* methods, both of which aim to improve the rate of convergence from sublinear to linear. These methods do not simply compute mini-batches of fixed size, nor do they compute full gradients in every iteration. Instead, they dynamically replace or incorporate new gradient information in order to construct a more reliable step with smaller variance than an SG step. For this reason, we refer to the methods along the horizontal axis as *noise reduction methods*. We discuss methods of this type in §5.

Along the second axis in Figure 3.3 are algorithms that, in a broad sense, attempt to overcome the adverse effects of high nonlinearity and ill-conditioning. For such algorithms, we use the term *second-order methods*, which encompasses a variety of strategies; see §6. We discuss well known


---

## Page 21

inexact Newton and quasi-Newton methods, as well as (generalized) Gauss-Newton methods [14, 141], the natural gradient method [5], and scaled gradient iterations [152, 54].

We caution that the schematic representation of methods presented in Figure 3.3 should not be taken too literally since it is not possible to truly organize algorithms so simply, or to include all methods along only two such axes. For example, one could argue that iterate averaging methods do not neatly belong in the category of second-order methods, even though we place them there, and one could argue that gradient methods with momentum [123] or acceleration [107, 108] do belong in this category, even though we discuss them separately in §7. Nevertheless, Figure 3.3 provides a useful road map as we describe and analyze a large collection of optimization methods of various forms and characteristics. Moreover, our two-dimensional roadmap is useful in that it suggests that optimization methods do not need to exist along the coordinate axes only; e.g., a batch Newton method is placed at the lower-right corner, and one may consider various combinations of second-order and noise reduction schemes.

## 4 Analyses of Stochastic Gradient Methods

In this section, we provide insights into the behavior of a stochastic gradient method (SG) by establishing its convergence properties and worst-case iteration complexity bounds. A preview of such properties were given in (3.10)–(3.11), but now we prove these and other interesting results in detail, all within the context of a generalized SG algorithm. We start by analyzing our SG algorithm when it is invoked to minimize a strongly convex objective function, where it is possible to establish a global rate of convergence to the optimal objective value. This is followed by analyses when our SG algorithm is employed to minimize a generic nonconvex objective. To emphasize the generality of the results proved in this section, we remark that the objective function under consideration could be the expected risk (3.4) or empirical risk (3.6); i.e., we refer to the objective function  $F : \mathbb{R}^d \rightarrow \mathbb{R}$ , which represents either

$$F(w) = \begin{cases} R(w) = \mathbb{E}[f(w; \xi)] \\ \text{or} \\ R_n(w) = \frac{1}{n} \sum_{i=1}^n f_i(w). \end{cases} \quad (4.1)$$

Our analyses apply equally to both objectives; the only difference lies in the way that one picks the stochastic gradient estimates in the method.<sup>3</sup>

We define our generalized SG method as Algorithm 4.1. The algorithm merely presumes that three computational tools exist: (i) a mechanism for generating a realization of a random variable  $\xi_k$  (with  $\{\xi_k\}$  representing a sequence of jointly independent random variables); (ii) given an iterate  $w_k \in \mathbb{R}^d$  and the realization of  $\xi_k$ , a mechanism for computing a stochastic vector  $g(w_k, \xi_k) \in \mathbb{R}^d$ ; and (iii) given an iteration number  $k \in \mathbb{N}$ , a mechanism for computing a scalar stepsize  $\alpha_k > 0$ .

---

<sup>3</sup>Picking samples uniformly from a finite training set, replacing them into the set for each iteration, corresponds to sampling from a discrete distribution giving equal weight to every sample. In this case, the SG algorithm in this section optimizes the empirical risk  $F = R_n$ . Alternatively, picking samples in each iteration according to the distribution  $P$ , the SG algorithm optimizes the expected risk  $F = R$ . One could also imagine picking samples *without replacement* until one exhausts a finite training set. In this case, the SG algorithm here can be viewed as optimizing either  $R_n$  or  $R$ , but only until the training set is exhausted. After that point, our analyses no longer apply. Generally speaking, the analyses of such *incremental algorithms* often requires specialized techniques [15, 72].


---

## Page 22

---

**Algorithm 4.1** Stochastic Gradient (SG) Method

---

```
1: Choose an initial iterate  $w_1$ .
2: for  $k = 1, 2, \dots$  do
3:   Generate a realization of the random variable  $\xi_k$ .
4:   Compute a stochastic vector  $g(w_k, \xi_k)$ .
5:   Choose a stepsize  $\alpha_k > 0$ .
6:   Set the new iterate as  $w_{k+1} \leftarrow w_k - \alpha_k g(w_k, \xi_k)$ .
7: end for
```

---

The generality of Algorithm 4.1 can be seen in various ways. First, the value of the random variable  $\xi_k$  need only be viewed as a seed for generating a stochastic direction; as such, a realization of it may represent the choice of a single training sample as in the simple SG method stated as (3.7), or may represent a set of samples as in the mini-batch SG method (3.12). Second,  $g(w_k, \xi_k)$  could represent a stochastic gradient—i.e., an unbiased estimator of  $\nabla F(w_k)$ , as in the classical method of Robbins and Monro [130]—or it could represent a stochastic Newton or quasi-Newton direction; see §6. That is, our analyses cover the choices

$$g(w_k, \xi_k) = \begin{cases} \nabla f(w_k; \xi_k) \\ \frac{1}{n_k} \sum_{i=1}^{n_k} \nabla f(w_k; \xi_{k,i}) \\ H_k \frac{1}{n_k} \sum_{i=1}^{n_k} \nabla f(w_k; \xi_{k,i}), \end{cases} \quad (4.2)$$

where, for all  $k \in \mathbb{N}$ , one has flexibility in the choice of mini-batch size  $n_k$  and symmetric positive definite scaling matrix  $H_k$ . No matter what choice is made, we shall come to see that all of our theoretical results hold as long as the expected angle between  $g(w_k, \xi_k)$  and  $\nabla F(w_k)$  is sufficiently positive. Third, Algorithm 4.1 allows various choices of the stepsize sequence  $\{\alpha_k\}$ . Our analyses focus on two choices, one involving a fixed stepsize and one involving diminishing stepsizes, as both are interesting in theory and in practice. Finally, we note that Algorithm 4.1 also covers active learning techniques in which the iterate  $w_k$  influences the sample selection.<sup>4</sup>

Notwithstanding all of this generality, we henceforth refer to Algorithm 4.1 as *SG*. The particular instance (3.7) will be referred to as *simple SG* or *basic SG*, whereas the instance (3.12) will be referred to as *mini-batch SG*.

Beyond our convergence and complexity analyses, a complete appreciation for the properties of SG is not possible without highlighting its theoretical advantages over batch methods in terms of computational complexity. Thus, we include in section §4.4 a discussion of the trade-offs between rate of convergence and computational effort among prototypical stochastic and batch methods for large-scale learning.

---

<sup>4</sup>We have assumed that the elements of the random variable sequence  $\{\xi_k\}$  are independent in order to avoid requiring certain machinery from the analyses of stochastic processes. Viewing  $\xi_k$  as a seed instead of a sample during iteration  $k$  makes this restriction minor. However, it is worthwhile to mention that all of the results in this section still hold if, instead,  $\{\xi_k\}$  forms an adapted (non-anticipating) stochastic process and expectations taken with respect to  $\xi_k$  are replaced by expectations taken with respect to the conditional distribution of  $\xi_k$  given  $\{\xi_1, \dots, \xi_{k-1}\}$ .


---

## Page 23

## 4.1 Two Fundamental Lemmas

Our approach for establishing convergence guarantees for SG is built upon an assumption of smoothness of the objective function. (Alternative foundations are possible; see Appendix A.) This, and an assumption about the first and second moments of the stochastic vectors  $\{g(w_k, \xi_k)\}$  lead to two fundamental lemmas from which all of our results will be derived.

Our first assumption is formally stated as the following. Recall that, as already mentioned in (4.1),  $F$  can stand for either expected or empirical risk.

**Assumption 4.1 (Lipschitz-continuous objective gradients).** *The objective function  $F : \mathbb{R}^d \rightarrow \mathbb{R}$  is continuously differentiable and the gradient function of  $F$ , namely,  $\nabla F : \mathbb{R}^d \rightarrow \mathbb{R}^d$ , is Lipschitz continuous with Lipschitz constant  $L > 0$ , i.e.,*

$$\|\nabla F(w) - \nabla F(\bar{w})\|_2 \leq L\|w - \bar{w}\|_2 \text{ for all } \{w, \bar{w}\} \subset \mathbb{R}^d.$$

Intuitively, Assumption 4.1 ensures that the gradient of  $F$  does not change arbitrarily quickly with respect to the parameter vector. Such an assumption is essential for convergence analyses of most gradient-based methods; without it, the gradient would not provide a good indicator for how far to move to decrease  $F$ . An important consequence of Assumption 4.1 is that

$$F(w) \leq F(\bar{w}) + \nabla F(\bar{w})^T (w - \bar{w}) + \frac{1}{2}L\|w - \bar{w}\|_2^2 \text{ for all } \{w, \bar{w}\} \subset \mathbb{R}^d. \quad (4.3)$$

This inequality is proved in Appendix B, but note that it also follows immediately if  $F$  is twice continuously differentiable and the Hessian function  $\nabla^2 F : \mathbb{R}^d \rightarrow \mathbb{R}^{d \times d}$  satisfies  $\|\nabla^2 F(w)\|_2 \leq L$  for all  $w \in \mathbb{R}^d$ .

Under Assumption 4.1 alone, we obtain the following lemma. In the result, we use  $\mathbb{E}_{\xi_k}[\cdot]$  to denote an expected value taken with respect to the distribution of the random variable  $\xi_k$  given  $w_k$ . Therefore,  $\mathbb{E}_{\xi_k}[F(w_{k+1})]$  is a meaningful quantity since  $w_{k+1}$  depends on  $\xi_k$  through the update in Step 6 of Algorithm 4.1.

**Lemma 4.2.** *Under Assumption 4.1, the iterates of SG (Algorithm 4.1) satisfy the following inequality for all  $k \in \mathbb{N}$ :*

$$\mathbb{E}_{\xi_k}[F(w_{k+1})] - F(w_k) \leq -\alpha_k \nabla F(w_k)^T \mathbb{E}_{\xi_k}[g(w_k, \xi_k)] + \frac{1}{2}\alpha_k^2 L \mathbb{E}_{\xi_k}[\|g(w_k, \xi_k)\|_2^2]. \quad (4.4)$$

*Proof.* By Assumption 4.1, the iterates generated by SG satisfy

$$\begin{aligned} F(w_{k+1}) - F(w_k) &\leq \nabla F(w_k)^T (w_{k+1} - w_k) + \frac{1}{2}L\|w_{k+1} - w_k\|_2^2 \\ &\leq -\alpha_k \nabla F(w_k)^T g(w_k, \xi_k) + \frac{1}{2}\alpha_k^2 L \|g(w_k, \xi_k)\|_2^2. \end{aligned}$$

Taking expectations in these inequalities with respect to the distribution of  $\xi_k$ , and noting that  $w_{k+1}$ —but not  $w_k$ —depends on  $\xi_k$ , we obtain the desired bound.  $\square$

This lemma shows that, regardless of how SG arrived at  $w_k$ , the expected decrease in the objective function yielded by the  $k$ th step is bounded above by a quantity involving: (i) the expected directional derivative of  $F$  at  $w_k$  along  $-g(w_k, \xi_k)$  and (ii) the second moment of  $g(w_k, \xi_k)$ . For example, if  $g(w_k, \xi_k)$  is an unbiased estimate of  $\nabla F(w_k)$ , then it follows from Lemma 4.2 that

$$\mathbb{E}_{\xi_k}[F(w_{k+1})] - F(w_k) \leq -\alpha_k \|\nabla F(w_k)\|_2^2 + \frac{1}{2}\alpha_k^2 L \mathbb{E}_{\xi_k}[\|g(w_k, \xi_k)\|_2^2]. \quad (4.5)$$


---

## Page 24

We shall see that convergence of SG is guaranteed as long as the stochastic directions and stepsizes are chosen such that the right-hand side of (4.4) is bounded above by a *deterministic* quantity that asymptotically ensures sufficient descent in  $F$ . One can ensure this in part by stating additional requirements on the first and second moments of the stochastic directions  $\{g(w_k, \xi_k)\}$ . In particular, in order to limit the harmful effect of the last term in (4.5), we restrict the variance of  $g(w_k, \xi_k)$ , i.e.,

$$\mathbb{V}_{\xi_k}[g(w_k, \xi_k)] := \mathbb{E}_{\xi_k}[\|g(w_k, \xi_k)\|_2^2] - \|\mathbb{E}_{\xi_k}[g(w_k, \xi_k)]\|_2^2. \quad (4.6)$$

**Assumption 4.3 (First and second moment limits).** *The objective function and SG (Algorithm 4.1) satisfy the following:*

- (a) *The sequence of iterates  $\{w_k\}$  is contained in an open set over which  $F$  is bounded below by a scalar  $F_{\inf}$ .*
- (b) *There exist scalars  $\mu_G \geq \mu > 0$  such that, for all  $k \in \mathbb{N}$ ,*

$$\nabla F(w_k)^T \mathbb{E}_{\xi_k}[g(w_k, \xi_k)] \geq \mu \|\nabla F(w_k)\|_2^2 \quad \text{and} \quad (4.7a)$$

$$\|\mathbb{E}_{\xi_k}[g(w_k, \xi_k)]\|_2 \leq \mu_G \|\nabla F(w_k)\|_2. \quad (4.7b)$$

- (c) *There exist scalars  $M \geq 0$  and  $M_V \geq 0$  such that, for all  $k \in \mathbb{N}$ ,*

$$\mathbb{V}_{\xi_k}[g(w_k, \xi_k)] \leq M + M_V \|\nabla F(w_k)\|_2^2. \quad (4.8)$$

The first condition, Assumption 4.3(a), merely requires the objective function to be bounded below over the region explored by the algorithm. The second requirement, Assumption 4.3(b), states that, in expectation, the vector  $-g(w_k, \xi_k)$  is a direction of sufficient descent for  $F$  from  $w_k$  with a norm comparable to the norm of the gradient. The properties in this requirement hold immediately with  $\mu_G = \mu = 1$  if  $g(w_k, \xi_k)$  is an unbiased estimate of  $\nabla F(w_k)$ , and are maintained if such an unbiased estimate is multiplied by a positive definite matrix  $H_k$  that is conditionally uncorrelated with  $g(w_k, \xi_k)$  given  $w_k$  and whose eigenvalues lie in a fixed positive interval for all  $k \in \mathbb{N}$ . The third requirement, Assumption 4.3(c), states that the variance of  $g(w_k, \xi_k)$  is restricted, but in a relatively minor manner. For example, if  $F$  is a convex quadratic function, then the variance is allowed to be nonzero at any stationary point for  $F$  and is allowed to grow quadratically in any direction.

All together, Assumption 4.3, combined with the definition (4.6), requires that the second moment of  $g(w_k, \xi_k)$  satisfies

$$\mathbb{E}_{\xi_k}[\|g(w_k, \xi_k)\|_2^2] \leq M + M_G \|\nabla F(w_k)\|_2^2 \quad \text{with} \quad M_G := M_V + \mu_G^2 \geq \mu^2 > 0. \quad (4.9)$$

In fact, all of our analyses in this section hold if this bound on the second moment were to be assumed directly. (We have stated Assumption 4.3 in the form above merely to facilitate our discussion in §5.)

The following lemma builds on Lemma 4.2 under the additional conditions now set forth in Assumption 4.3.

**Lemma 4.4.** *Under Assumptions 4.1 and 4.3, the iterates of SG (Algorithm 4.1) satisfy the following inequalities for all  $k \in \mathbb{N}$ :*

$$\mathbb{E}_{\xi_k}[F(w_{k+1})] - F(w_k) \leq -\mu \alpha_k \|\nabla F(w_k)\|_2^2 + \frac{1}{2} \alpha_k^2 L \mathbb{E}_{\xi_k}[\|g(w_k, \xi_k)\|_2^2] \quad (4.10a)$$

$$\leq -(\mu - \frac{1}{2} \alpha_k L M_G) \alpha_k \|\nabla F(w_k)\|_2^2 + \frac{1}{2} \alpha_k^2 L M. \quad (4.10b)$$


---

## Page 25

*Proof.* By Lemma 4.2 and (4.7a), it follows that

$$\begin{aligned}\mathbb{E}_{\xi_k}[F(w_{k+1})] - F(w_k) &\leq -\alpha_k \nabla F(w_k)^T \mathbb{E}_{\xi_k}[g(w_k, \xi_k)] + \frac{1}{2} \alpha_k^2 L \mathbb{E}_{\xi_k}[\|g(w_k, \xi_k)\|_2^2] \\ &\leq -\mu \alpha_k \|\nabla F(w_k)\|_2^2 + \frac{1}{2} \alpha_k^2 L \mathbb{E}_{\xi_k}[\|g(w_k, \xi_k)\|_2^2],\end{aligned}$$

which is (4.10a). Assumption 4.3, giving (4.9), then yields (4.10b).  $\square$

As mentioned, this lemma reveals that regardless of how the method arrived at the iterate  $w_k$ , the optimization process continues in a *Markovian* manner in the sense that  $w_{k+1}$  is a random variable that depends only on the iterate  $w_k$ , the seed  $\xi_k$ , and the stepsize  $\alpha_k$  and not on any past iterates. This can be seen in the fact that the difference  $\mathbb{E}_{\xi_k}[F(w_{k+1})] - F(w_k)$  is bounded above by a deterministic quantity. Note also that the first term in (4.10b) is strictly negative for small  $\alpha_k$  and suggests a decrease in the objective function by a magnitude proportional to  $\|\nabla F(w_k)\|_2^2$ . However, the second term in (4.10b) could be large enough to allow the objective value to increase. Balancing these terms is critical in the design of SG methods.

## 4.2 SG for Strongly Convex Objectives

The most benign setting for analyzing the SG method is in the context of minimizing a strongly convex objective function. For the reasons described in Inset 4.2, when not considering a generic nonconvex objective  $F$ , we focus on the strongly convex case and only briefly mention the (not strongly) convex case in certain occasions.

### Inset 4.2: Perspectives on SG Analyses

All of the convergence rate and complexity results presented in this paper relate to the minimization of *strongly convex* functions. This is in contrast with a large portion of the literature on optimization methods for machine learning, in which much effort is placed to strengthen convergence guarantees for methods applied to functions that are convex, but not strongly convex. We have made this choice for a few reasons. First, it leads to a focus on results that are relevant to actual machine learning practice, since in many situations when a convex model is employed—such as in logistic regression—it is often regularized by a strongly convex function to facilitate the solution process. Second, there exist a variety of situations in which the objective function is not globally (strongly) convex, but is so in the neighborhood of local minimizers, meaning that our results can represent the behavior of the algorithm in such regions of the search space. Third, one can argue that related results when minimizing non-strongly convex models can be derived as extensions of the results presented here [3], making our analyses a starting point for deriving a more general theory.

We have also taken a pragmatic approach in the types of convergence guarantees that we provide. In particular, in our analyses, we focus on results that reveal the properties of SG iterates *in expectation*. The stochastic approximation literature, on the other hand, often relies on martingale techniques to establish *almost sure convergence* [66, 131] under the same assumptions [21]. For our purposes, we omit these complications since, in our view, they do not provide significant additional insights into the forces driving convergence of the method.

We formalize a strong convexity assumption as the following.


---

## Page 26

**Assumption 4.5 (Strong convexity).** *The objective function  $F : \mathbb{R}^d \rightarrow \mathbb{R}$  is strongly convex in that there exists a constant  $c > 0$  such that*

$$F(\bar{w}) \geq F(w) + \nabla F(w)^T (\bar{w} - w) + \frac{1}{2} c \|\bar{w} - w\|_2^2 \text{ for all } (\bar{w}, w) \in \mathbb{R}^d \times \mathbb{R}^d. \quad (4.11)$$

Hence,  $F$  has a unique minimizer, denoted as  $w_* \in \mathbb{R}^d$  with  $F_* := F(w_*)$ .

A useful fact from convex analysis (proved in Appendix B) is that, under Assumption 4.5, one can bound the optimality gap at a given point in terms of the squared  $\ell_2$ -norm of the gradient of the objective at that point:

$$2c(F(w) - F_*) \leq \|\nabla F(w)\|_2^2 \text{ for all } w \in \mathbb{R}^d. \quad (4.12)$$

We use this inequality in several proofs. We also observe that, from (4.3) and (4.11), the constants in Assumptions 4.1 and 4.5 must satisfy  $c \leq L$ .

We now state our first convergence theorem for SG, describing its behavior when minimizing a strongly convex objective function when employing a fixed stepsize. In this case, it will not be possible to prove convergence to the solution, but only to a neighborhood of the optimal value. (Intuitively, this limitation should be clear from (4.10b) since the first term on the right-hand side decreases in magnitude as the solution is approached—i.e., as  $\nabla F(w_k)$  tends to zero—but the last term remains constant. Thus, after some point, a reduction in the objective cannot be expected.) We use  $\mathbb{E}[\cdot]$  to denote an expected value taken with respect to the joint distribution of all random variables. For example, since  $w_k$  is completely determined by the realizations of the independent random variables  $\{\xi_1, \xi_2, \dots, \xi_{k-1}\}$ , the *total expectation* of  $F(w_k)$  for any  $k \in \mathbb{N}$  can be taken as

$$\mathbb{E}[F(w_k)] = \mathbb{E}_{\xi_1} \mathbb{E}_{\xi_2} \dots \mathbb{E}_{\xi_{k-1}} [F(w_k)].$$

The theorem shows that if the stepsize is not too large, then, in expectation, the sequence of function values  $\{F(w_k)\}$  converges near the optimal value.

**Theorem 4.6 (Strongly Convex Objective, Fixed Stepsize).** *Under Assumptions 4.1, 4.3, and 4.5 (with  $F_{\text{inf}} = F_*$ ), suppose that the SG method (Algorithm 4.1) is run with a fixed stepsize,  $\alpha_k = \bar{\alpha}$  for all  $k \in \mathbb{N}$ , satisfying*

$$0 < \bar{\alpha} \leq \frac{\mu}{LM_G}. \quad (4.13)$$

Then, the expected optimality gap satisfies the following inequality for all  $k \in \mathbb{N}$ :

$$\begin{aligned} \mathbb{E}[F(w_k) - F_*] &\leq \frac{\bar{\alpha}LM}{2c\mu} + (1 - \bar{\alpha}c\mu)^{k-1} \left( F(w_1) - F_* - \frac{\bar{\alpha}LM}{2c\mu} \right) \\ &\xrightarrow{k \rightarrow \infty} \frac{\bar{\alpha}LM}{2c\mu}. \end{aligned} \quad (4.14)$$

*Proof.* Using Lemma 4.4 with (4.13) and (4.12), we have for all  $k \in \mathbb{N}$  that

$$\begin{aligned} \mathbb{E}_{\xi_k} [F(w_{k+1})] - F(w_k) &\leq -(\mu - \frac{1}{2}\bar{\alpha}LM_G)\bar{\alpha}\|\nabla F(w_k)\|_2^2 + \frac{1}{2}\bar{\alpha}^2LM \\ &\leq -\frac{1}{2}\bar{\alpha}\mu\|\nabla F(w_k)\|_2^2 + \frac{1}{2}\bar{\alpha}^2LM \\ &\leq -\bar{\alpha}c\mu(F(w_k) - F_*) + \frac{1}{2}\bar{\alpha}^2LM. \end{aligned}$$


---

## Page 27

Subtracting  $F_*$  from both sides, taking total expectations, and rearranging, this yields

$$\mathbb{E}[F(w_{k+1}) - F_*] \leq (1 - \bar{\alpha}c\mu)\mathbb{E}[F(w_k) - F_*] + \frac{1}{2}\bar{\alpha}^2 LM.$$

Subtracting the constant  $\bar{\alpha}LM/(2c\mu)$  from both sides, one obtains

$$\begin{aligned} \mathbb{E}[F(w_{k+1}) - F_*] - \frac{\bar{\alpha}LM}{2c\mu} &\leq (1 - \bar{\alpha}c\mu)\mathbb{E}[F(w_k) - F_*] + \frac{\bar{\alpha}^2 LM}{2} - \frac{\bar{\alpha}LM}{2c\mu} \\ &= (1 - \bar{\alpha}c\mu)\left(\mathbb{E}[F(w_k) - F_*] - \frac{\bar{\alpha}LM}{2c\mu}\right). \end{aligned} \quad (4.15)$$

Observe that (4.15) is a contraction inequality since, by (4.13) and (4.9),

$$0 < \bar{\alpha}c\mu \leq \frac{c\mu^2}{LM_G} \leq \frac{c\mu^2}{L\mu^2} = \frac{c}{L} \leq 1. \quad (4.16)$$

The result thus follows by applying (4.15) repeatedly through iteration  $k \in \mathbb{N}$ .  $\square$

If  $g(w_k, \xi_k)$  is an unbiased estimate of  $\nabla F(w_k)$ , then  $\mu = 1$ , and if there is no noise in  $g(w_k, \xi_k)$ , then we may presume that  $M_G = 1$  (due to (4.9)). In this case, (4.13) reduces to  $\bar{\alpha} \in (0, 1/L]$ , a classical stepsize requirement of interest for a steepest descent method.

Theorem 4.6 illustrates the interplay between the stepsizes and bound on the variance of the stochastic directions. If there were no noise in the gradient computation or if noise were to decay with  $\|\nabla F(w_k)\|_2^2$  (i.e., if  $M = 0$  in (4.8) and (4.9)), then one can obtain linear convergence to the optimal value. This is a standard result for the full gradient method with a sufficiently small positive stepsize. On the other hand, when the gradient computation is noisy, one clearly loses this property. One can still use a fixed stepsize and be sure that the expected objective values will converge linearly to a neighborhood of the optimal value, but, after some point, the noise in the gradient estimates prevent further progress; recall Example 3.1. It is apparent from (4.14) that selecting a smaller stepsize worsens the contraction constant in the convergence rate, but allows one to arrive closer to the optimal value.

These observations provide a foundation for a strategy often employed in practice by which SG is run with a fixed stepsize, and, if progress appears to stall, a smaller stepsize is selected and the process is repeated. A straightforward instance of such an approach can be motivated with the following sketch. Suppose that  $\alpha_1 \in (0, \frac{\mu}{LM_G}]$  is chosen as in (4.13) and the SG method is run with this stepsize from iteration  $k_1 = 1$  until iteration  $k_2$ , where  $w_{k_2}$  is the first iterate at which the expected suboptimality gap is smaller than twice the asymptotic value in (4.14), i.e.,  $\mathbb{E}[F(w_{k_2}) - F_*] \leq 2F_{\alpha_1}$ , where  $F_{\alpha} := \frac{\alpha LM}{2c\mu}$ . Suppose further that, at this point, the stepsize is halved and the process is repeated; see Figure 4.1. This leads to the stepsize schedule  $\{\alpha_{r+1}\} = \{\alpha_1 2^{-r}\}$ , index sequence  $\{k_r\}$ , and bound sequence  $\{F_{\alpha_r}\} = \{\frac{\alpha_r LM}{2c\mu}\} \searrow 0$  such that, for all  $r \in \{2, 3, \dots\}$ ,

$$\mathbb{E}[F(w_{k_{r+1}}) - F_*] \leq 2F_{\alpha_r} \quad \text{where} \quad \mathbb{E}[F(w_{k_r}) - F_*] \approx 2F_{\alpha_{r-1}} = 4F_{\alpha_r}. \quad (4.17)$$

In this manner, the expected suboptimality gap converges to zero.

However, this does *not* occur by halving the stepsize in every iteration, but only once the gap itself has been cut in half from a previous threshold. To see what is the appropriate *effective rate*


---

## Page 28

of stepsize decrease, we may invoke Theorem 4.6, from which it follows that to achieve the first bound in (4.17) one needs

$$\begin{aligned} (1 - \alpha_r c\mu)^{(k_{r+1} - k_r)} (4F_{\alpha_r} - F_{\alpha_r}) &\leq F_{\alpha_r} \\ \implies k_{r+1} - k_r &\geq \frac{\log(1/3)}{\log(1 - \alpha_r c\mu)} \approx \frac{\log(3)}{\alpha_r c\mu} = \mathcal{O}(2^r). \end{aligned} \quad (4.18)$$

In other words, each time the stepsize is cut in half, double the number of iterations are required. This is a *sublinear* rate of stepsize decrease—e.g., if  $\{k_r\} = \{2^{r-1}\}$ , then  $\alpha_k = \alpha_1/k$  for all  $k \in \{2^r\}$ —which, from  $\{F_{\alpha_r}\} = \{\frac{\alpha_r LM}{2c\mu}\}$  and (4.17), means that a sublinear convergence rate of the suboptimality gap is achieved.

![Figure 4.1: A graph showing the expected suboptimality gap E[F(w_k)] versus the number of iterations k. The curve starts at a high value and decreases towards the asymptotic limit F*. Three horizontal dashed lines represent the asymptotic limits for step sizes 2α, α, and α/2. The step size 2α is shown in red, α in green, and α/2 in brown. Points A and B are on the curve, and A' and B' are on the horizontal lines. The vertical distance from A to A' is labeled 2α, and from B to B' is labeled α/2. The horizontal distance from A to B is labeled α. The graph illustrates that halving the step size requires doubling the number of iterations to achieve the same reduction in the suboptimality gap.](9106a02a0dd21fa6f937aaf400af1cc5_4_img.webp)

Fig. 4.1: Depiction of the strategy of halving the stepsize  $\alpha$  when the expected suboptimality gap is smaller than twice the asymptotic limit  $F_\alpha$ . In the figure, the segment  $B-B'$  has one third of the length of  $A-A'$ . This is the amount of decrease that must be made in the exponential term in (4.14) by raising the contraction factor to the power of the number of steps during which one maintains a given constant stepsize; see (4.18). Since the contraction factor is  $(1 - \alpha c\mu)$ , the number of steps must be proportional to  $\alpha$ . Therefore, whenever the stepsize is halved, one must maintain it twice as long. Overall, doubling the number of iterations halves the suboptimality gap each time, yielding an effective rate of  $\mathcal{O}(1/k)$ .

In fact, these conclusions can be obtained in a more rigorous manner that also allows more flexibility in the choice of stepsize sequence. The following result harks back to the seminal work of Robbins and Monro [130], where the stepsize requirement takes the form

$$\sum_{k=1}^{\infty} \alpha_k = \infty \quad \text{and} \quad \sum_{k=1}^{\infty} \alpha_k^2 < \infty. \quad (4.19)$$

**Theorem 4.7 (Strongly Convex Objective, Diminishing Stepsizes).** *Under Assumptions 4.1, 4.3, and 4.5 (with  $F_{\inf} = F_*$ ), suppose that the SG method (Algorithm 4.1) is run with a stepsize sequence such that, for all  $k \in \mathbb{N}$ ,*

$$\alpha_k = \frac{\beta}{\gamma + k} \quad \text{for some } \beta > \frac{1}{c\mu} \quad \text{and } \gamma > 0 \quad \text{such that } \alpha_1 \leq \frac{\mu}{LM_G}. \quad (4.20)$$


---

## Page 29

Then, for all  $k \in \mathbb{N}$ , the expected optimality gap satisfies

$$\mathbb{E}[F(w_k) - F_*] \leq \frac{\nu}{\gamma + k}, \quad (4.21)$$

where

$$\nu := \max \left\{ \frac{\beta^2 LM}{2(\beta c \mu - 1)}, (\gamma + 1)(F(w_1) - F_*) \right\}. \quad (4.22)$$

*Proof.* By (4.20), the inequality  $\alpha_k LM_G \leq \alpha_1 LM_G \leq \mu$  holds for all  $k \in \mathbb{N}$ . Hence, along with Lemma 4.4 and (4.12), one has for all  $k \in \mathbb{N}$  that

$$\begin{aligned} \mathbb{E}_{\xi_k}[F(w_{k+1})] - F(w_k) &\leq -(\mu - \frac{1}{2}\alpha_k LM_G)\alpha_k \|\nabla F(w_k)\|_2^2 + \frac{1}{2}\alpha_k^2 LM \\ &\leq -\frac{1}{2}\alpha_k \mu \|\nabla F(w_k)\|_2^2 + \frac{1}{2}\alpha_k^2 LM \\ &\leq -\alpha_k c \mu (F(w_k) - F(w_*)) + \frac{1}{2}\alpha_k^2 LM. \end{aligned}$$

Subtracting  $F_*$  from both sides, taking total expectations, and rearranging, this yields

$$\mathbb{E}[F(w_{k+1}) - F_*] \leq (1 - \alpha_k c \mu) \mathbb{E}[F(w_k) - F_*] + \frac{1}{2}\alpha_k^2 LM. \quad (4.23)$$

We now prove (4.21) by induction. First, the definition of  $\nu$  ensures that it holds for  $k = 1$ . Then, assuming (4.21) holds for some  $k \geq 1$ , it follows from (4.23) that

$$\begin{aligned} \mathbb{E}[F(w_{k+1}) - F_*] &\leq \left(1 - \frac{\beta c \mu}{\hat{k}}\right) \frac{\nu}{\hat{k}} + \frac{\beta^2 LM}{2\hat{k}^2} \quad (\text{with } \hat{k} := \gamma + k) \\ &= \left(\frac{\hat{k} - \beta c \mu}{\hat{k}^2}\right) \nu + \frac{\beta^2 LM}{2\hat{k}^2} \\ &= \left(\frac{\hat{k} - 1}{\hat{k}^2}\right) \nu - \underbrace{\left(\frac{\beta c \mu - 1}{\hat{k}^2}\right) \nu + \frac{\beta^2 LM}{2\hat{k}^2}}_{\text{nonpositive by the definition of } \nu} \leq \frac{\nu}{\hat{k} + 1}, \end{aligned}$$

where the last inequality follows because  $\hat{k}^2 \geq (\hat{k} + 1)(\hat{k} - 1)$ .  $\square$

Let us now remark on what can be learned from Theorems 4.6 and 4.7.

**Role of Strong Convexity** Observe the crucial role played by the strong convexity parameter  $c > 0$ , the positivity of which is needed to argue that (4.15) and (4.23) contract the expected optimality gap. However, the strong convexity constant impacts the stepsizes in different ways in Theorems 4.6 and 4.7. In the case of constant stepsizes, the possible values of  $\bar{\alpha}$  are constrained by the upper bound (4.13) that does not depend on  $c$ . In the case of diminishing stepsizes, the initial stepsize  $\alpha_1$  is subject to the same upper bound (4.20), but the stepsize parameter  $\beta$  must be larger than  $1/(c\mu)$ . This additional requirement is critical to ensure the  $\mathcal{O}(1/k)$  convergence rate. How critical? Consider, e.g., [105] in which the authors provide a simple example (with unbiased gradient estimates and  $\mu = 1$ ) involving the minimization of a deterministic quadratic function *with only one optimization variable* in which  $c$  is overestimated, which results in  $\beta$  being underestimated. In the example, even after  $10^9$  iterations, the distance to the solution remains greater than  $10^{-2}$ .


---

## Page 30

**Role of the Initial Point** Also observe the role played by the initial point, which determines the initial optimality gap, namely,  $F(w_1) - F_*$ . When using a fixed stepsize, the initial gap appears with an exponentially decreasing factor; see (4.14). In the case of diminishing stepsizes, the gap appears prominently in the second term defining  $\nu$  in (4.22). However, with an appropriate initialization phase, one can easily diminish the role played by this term.<sup>5</sup> For example, suppose that one begins by running SG with a fixed stepsize  $\bar{\alpha}$  until one (approximately) obtains a point, call it  $w_1$ , with  $F(w_1) - F_* \leq \bar{\alpha}LM/(2c\mu)$ . A guarantee for this bound can be argued from (4.14). Starting here with  $\alpha_1 = \bar{\alpha}$ , the choices for  $\beta$  and  $\gamma$  in Theorem 4.7 yield

$$(\gamma + 1)\mathbb{E}[F(w_1) - F_*] \leq \beta\alpha_1^{-1}\frac{\alpha_1LM}{2c\mu} = \frac{\beta LM}{2c\mu} < \frac{\beta^2 LM}{2(\beta c\mu - 1)},$$

meaning that the value for  $\nu$  is dominated by the first term in (4.22).

On a related note, we claim that for practical purposes the initial stepsize should be chosen as large as allowed, i.e.,  $\alpha_1 = \mu/(LM_G)$ . Given this choice of  $\alpha_1$ , the best asymptotic regime with decreasing stepsizes (4.21) is achieved by making  $\nu$  as small as possible. Since we have argued that only the first term matters in the definition of  $\nu$ , this leads to choosing  $\beta = 2/(c\mu)$ . Under these conditions, one has

$$\nu = \frac{\beta^2 LM}{2(\beta c\mu - 1)} = \frac{2}{\mu^2} \left(\frac{L}{c}\right) \left(\frac{M}{c}\right). \quad (4.24)$$

We shall see the (potentially large) ratios  $L/c$  and  $M/c$  arise again later.

**Trade-Offs of (Mini-)Batching** As a final observation about what can be learned from Theorems 4.6 and 4.7, let us take a moment to compare the theoretical performance of two fundamental algorithms—the simple SG iteration (3.7) and the mini-batch SG iteration (3.12)—when these results are applied for minimizing empirical risk, i.e., when  $F = R_n$ . This provides a glimpse into how such results can be used to compare algorithms in terms of their computational trade-offs.

The most elementary instance of our SG algorithm is simple SG, which, as we have seen, consists of picking a random sample index  $i_k$  at each iteration and computing

$$g(w_k, \xi_k) = \nabla f_{i_k}(w_k). \quad (4.25)$$

By contrast, instead of picking a single sample, mini-batch SG consists of randomly selecting a subset  $\mathcal{S}_k$  of the sample indices and computing

$$g(w_k, \xi_k) = \frac{1}{|\mathcal{S}_k|} \sum_{i \in \mathcal{S}_k} \nabla f_i(w_k). \quad (4.26)$$

To compare these methods, let us assume for simplicity that we employ the same number of samples in each iteration so that the mini-batches are of constant size, i.e.,  $|\mathcal{S}_k| = n_{\text{mb}}$ . There are then two distinct regimes to consider, namely, when  $n_{\text{mb}} \ll n$  and when  $n_{\text{mb}} \approx n$ . Our goal here is to use the results of Theorems 4.6 and 4.7 to show that, in the former scenario, the theoretical benefit of mini-batching can appear to be somewhat ambiguous, meaning that one must leverage

---

<sup>5</sup>In fact, the bound (4.21) slightly overstates the asymptotic influence of the initial optimality gap. Applying Chung’s lemma [36] to the contraction equation (4.23) shows that the first term in the definition of  $\nu$  effectively determines the asymptotic convergence rate of  $\mathbb{E}[F(w_k) - F_*]$ .


---

## Page 31

certain computational tools to benefit from mini-batching in practice. As for the scenario when  $n_{\text{mb}} \approx n$ , the comparison is more complex due to a trade-off between per-iteration costs and overall convergence rate of the method (recall §3.3). We leave a more formal treatment of this scenario, specifically with the goals of large-scale machine learning in mind, for §4.4.

Suppose then that the mini-batch size is  $n_{\text{mb}} \ll n$ . The computation of the stochastic direction  $g(w_k, \xi_k)$  in (4.26) is clearly  $n_{\text{mb}}$  times more expensive than in (4.25). In return, the variance of the direction is reduced by a factor of  $1/n_{\text{mb}}$ . (See §5.2 for further discussion of this fact.) That is, with respect to our analysis, the constants  $M$  and  $M_V$  that appear in Assumption 4.3 (see (4.8)) are reduced by the same factor, becoming  $M/n_{\text{mb}}$  and  $M_V/n_{\text{mb}}$  for mini-batch SG. It is natural to ask whether this reduction of the variance pays for the higher per-iteration cost.

Consider, for instance, the case of employing a sufficiently small constant stepsize  $\bar{\alpha} > 0$ . For mini-batch SG, Theorem 4.6 leads to

$$\mathbb{E}[F(w_k) - F_*] \leq \frac{\bar{\alpha}LM}{2c\mu n_{\text{mb}}} + [1 - \bar{\alpha}c\mu]^{k-1} \left( F(w_1) - F_* - \frac{\bar{\alpha}LM}{2c\mu n_{\text{mb}}} \right).$$

Using the simple SG method with stepsize  $\bar{\alpha}/n_{\text{mb}}$  leads to a similar asymptotic gap:

$$\mathbb{E}[F(w_k) - F_*] \leq \frac{\bar{\alpha}LM}{2c\mu n_{\text{mb}}} + \left[ 1 - \frac{\bar{\alpha}c\mu}{n_{\text{mb}}} \right]^{k-1} \left( F(w_1) - F_* - \frac{\bar{\alpha}LM}{2c\mu n_{\text{mb}}} \right).$$

The worse contraction constant (indicated using square brackets) means that one needs to run  $n_{\text{mb}}$  times more iterations of the simple SG algorithm to obtain an equivalent optimality gap. That said, since the computation in a simple SG iteration is  $n_{\text{mb}}$  times cheaper, this amounts to effectively the same total computation as for the mini-batch SG method. A similar analysis employing the result of Theorem 4.7 can be performed when decreasing stepsizes are used.

These observations suggest that the methods can be comparable. However, an important consideration remains. In particular, the convergence theorems require that the initial stepsize be smaller than  $\mu/(LM_G)$ . Since (4.9) shows that  $M_G \geq \mu^2$ , the largest this stepsize can be is  $1/(\mu L)$ . Therefore, one cannot simply assume that the mini-batch SG method is allowed to employ a stepsize that is  $n_{\text{mb}}$  times larger than the one used by SG. In other words, one cannot always compensate for the higher per-iteration cost of a mini-batch SG method by selecting a larger stepsize.

One can, however, realize benefits of mini-batching in practice since it offers important opportunities for software optimization and parallelization; e.g., using sizeable mini-batches is often the only way to fully leverage a GPU processor. Dynamic mini-batch sizes can also be used as a substitute for decreasing stepsizes; see §5.2.

### 4.3 SG for General Objectives

As mentioned in our case study of deep neural networks in §2.2, many important machine learning models lead to nonconvex optimization problems, which are currently having a profound impact in practice. Analyzing the SG method when minimizing nonconvex objectives is more challenging than in the convex case since such functions may possess multiple local minima and other stationary points. Still, we show in this subsection that one can provide meaningful guarantees for the SG method in nonconvex settings.

Paralleling §4.2, we present two results—one for employing a fixed positive stepsize and one for diminishing stepsizes. We maintain the same assumptions about the stochastic directions  $g(w_k, \xi_k)$ ,


---

## Page 32

but do not assume convexity of  $F$ . As before, the results in this section still apply to a wide class of methods since  $g(w_k, \xi_k)$  could be defined as a (mini-batch) stochastic gradient or a Newton-like direction; recall (4.2).

Our first result describes the behavior of the sequence of gradients of  $F$  when fixed stepsizes are employed. Recall from Assumption 4.3 that the sequence of function values  $\{F(w_k)\}$  is assumed to be bounded below by a scalar  $F_{\inf}$ .

**Theorem 4.8 (Nonconvex Objective, Fixed Stepsize).** *Under Assumptions 4.1 and 4.3, suppose that the SG method (Algorithm 4.1) is run with a fixed stepsize,  $\alpha_k = \bar{\alpha}$  for all  $k \in \mathbb{N}$ , satisfying*

$$0 < \bar{\alpha} \leq \frac{\mu}{LM_G}. \quad (4.27)$$

*Then, the expected sum-of-squares and average-squared gradients of  $F$  corresponding to the SG iterates satisfy the following inequalities for all  $K \in \mathbb{N}$ :*

$$\mathbb{E} \left[ \sum_{k=1}^K \|\nabla F(w_k)\|_2^2 \right] \leq \frac{K\bar{\alpha}LM}{\mu} + \frac{2(F(w_1) - F_{\inf})}{\mu\bar{\alpha}} \quad (4.28a)$$

$$\begin{aligned} \text{and therefore } \mathbb{E} \left[ \frac{1}{K} \sum_{k=1}^K \|\nabla F(w_k)\|_2^2 \right] &\leq \frac{\bar{\alpha}LM}{\mu} + \frac{2(F(w_1) - F_{\inf})}{K\mu\bar{\alpha}} \\ &\xrightarrow{K \rightarrow \infty} \frac{\bar{\alpha}LM}{\mu}. \end{aligned} \quad (4.28b)$$

*Proof.* Taking the total expectation of (4.10b) and from (4.27),

$$\begin{aligned} \mathbb{E}[F(w_{k+1})] - \mathbb{E}[F(w_k)] &\leq -(\mu - \frac{1}{2}\bar{\alpha}LM_G)\bar{\alpha}\mathbb{E}[\|\nabla F(w_k)\|_2^2] + \frac{1}{2}\bar{\alpha}^2LM \\ &\leq -\frac{1}{2}\mu\bar{\alpha}\mathbb{E}[\|\nabla F(w_k)\|_2^2] + \frac{1}{2}\bar{\alpha}^2LM. \end{aligned}$$

Summing both sides of this inequality for  $k \in \{1, \dots, K\}$  and recalling Assumption 4.3(a) gives

$$F_{\inf} - F(w_1) \leq \mathbb{E}[F(w_{K+1})] - F(w_1) \leq -\frac{1}{2}\mu\bar{\alpha} \sum_{k=1}^K \mathbb{E}[\|\nabla F(w_k)\|_2^2] + \frac{1}{2}K\bar{\alpha}^2LM.$$

Rearranging yields (4.28a), and dividing further by  $K$  yields (4.28b).  $\square$

If  $M = 0$ , meaning that there is no noise or that noise reduces proportionally to  $\|\nabla F(w_k)\|_2^2$  (see equations (4.8) and (4.9)), then (4.28a) captures a classical result for the full gradient method applied to nonconvex functions, namely, that the sum of squared gradients remains finite, implying that  $\{\|\nabla F(w_k)\|_2\} \rightarrow 0$ . In the presence of noise (i.e.,  $M > 0$ ), Theorem 4.8 illustrates the interplay between the stepsize  $\bar{\alpha}$  and the variance of the stochastic directions. While one cannot bound the expected optimality gap as in the convex case, inequality (4.28b) bounds the average norm of the gradient of the objective function observed on  $\{w_k\}$  visited during the first  $K$  iterations. This quantity gets smaller when  $K$  increases, indicating that the SG method spends increasingly more time in regions where the objective function has a (relatively) small gradient. Moreover, the asymptotic result that one obtains from (4.28b) illustrates that noise in the gradients inhibits further progress, as in (4.14) for the convex case. The average norm of the gradients can be made


---

## Page 33

arbitrarily small by selecting a small stepsize, but doing so reduces the speed at which the norm of the gradient approaches its limiting distribution.

We now turn to the case when the SG method is applied to a nonconvex objective with a decreasing sequence of stepsizes satisfying the classical conditions (4.19). While not the strongest result that one can prove in this context—and, in fact, we prove a stronger result below—the following theorem is perhaps the easiest to interpret and remember. Hence, we state it first.

**Theorem 4.9 (Nonconvex Objective, Diminishing Stepsizes).** *Under Assumptions 4.1 and 4.3, suppose that the SG method (Algorithm 4.1) is run with a stepsize sequence satisfying (4.19). Then*

$$\liminf_{k \rightarrow \infty} \mathbb{E}[\|\nabla F(w_k)\|_2^2] = 0. \quad (4.29)$$

The proof of this theorem follows based on the results given in Theorem 4.10 below. A “lim inf” result of this type should be familiar to those knowledgeable of the nonlinear optimization literature. After all, such a result is all that can be shown for certain important methods, such as the nonlinear conjugate gradient method [114]. The intuition that one should gain from the statement of Theorem 4.9 is that, for the SG method with diminishing stepsizes, the expected gradient norms cannot stay bounded away from zero.

The following result characterizes more precisely the convergence property of SG.

**Theorem 4.10 (Nonconvex Objective, Diminishing Stepsizes).** *Under Assumptions 4.1 and 4.3, suppose that the SG method (Algorithm 4.1) is run with a stepsize sequence satisfying (4.19). Then, with  $A_K := \sum_{k=1}^K \alpha_k$ ,*

$$\lim_{K \rightarrow \infty} \mathbb{E} \left[ \sum_{k=1}^K \alpha_k \|\nabla F(w_k)\|_2^2 \right] < \infty \quad (4.30a)$$

$$\text{and therefore } \mathbb{E} \left[ \frac{1}{A_K} \sum_{k=1}^K \alpha_k \|\nabla F(w_k)\|_2^2 \right] \xrightarrow{K \rightarrow \infty} 0. \quad (4.30b)$$

*Proof.* The second condition in (4.19) ensures that  $\{\alpha_k\} \rightarrow 0$ , meaning that, without loss of generality, we may assume that  $\alpha_k LM_G \leq \mu$  for all  $k \in \mathbb{N}$ . Then, taking the total expectation of (4.10b),

$$\begin{aligned} \mathbb{E}[F(w_{k+1})] - \mathbb{E}[F(w_k)] &\leq -(\mu - \frac{1}{2}\alpha_k LM_G)\alpha_k \mathbb{E}[\|\nabla F(w_k)\|_2^2] + \frac{1}{2}\alpha_k^2 LM \\ &\leq -\frac{1}{2}\mu\alpha_k \mathbb{E}[\|\nabla F(w_k)\|_2^2] + \frac{1}{2}\alpha_k^2 LM. \end{aligned}$$

Summing both sides of this inequality for  $k \in \{1, \dots, K\}$  gives

$$F_{\inf} - \mathbb{E}[F(w_1)] \leq \mathbb{E}[F(w_{K+1})] - \mathbb{E}[F(w_1)] \leq -\frac{1}{2}\mu \sum_{k=1}^K \alpha_k \mathbb{E}[\|\nabla F(w_k)\|_2^2] + \frac{1}{2}LM \sum_{k=1}^K \alpha_k^2.$$

Dividing by  $\mu/2$  and rearranging the terms, we obtain

$$\sum_{k=1}^K \alpha_k \mathbb{E}[\|\nabla F(w_k)\|_2^2] \leq \frac{2(\mathbb{E}[F(w_1)] - F_{\inf})}{\mu} + \frac{LM}{\mu} \sum_{k=1}^K \alpha_k^2.$$

The second condition in (4.19) implies that the right-hand side of this inequality converges to a finite limit when  $K$  increases, proving (4.30a). Then, (4.30b) follows since the first condition in (4.19) ensures that  $A_K \rightarrow \infty$  as  $K \rightarrow \infty$ .  $\square$


---

## Page 34

Theorem 4.10 establishes results about a weighted sum-of-squares and a weighted average of squared gradients of  $F$  similar to those in Theorem 4.8. However, unlike (4.28b), the conclusion (4.30b) states that the weighted average norm of the squared gradients converges to zero even if the gradients are noisy, i.e., if  $M > 0$ . The fact that (4.30b) only specifies a property of a weighted average (with weights dictated by  $\{\alpha_k\}$ ) is only of minor importance since one can still conclude that the expected gradient norms cannot asymptotically stay far from zero.

We can now see that Theorem 4.9 is a direct consequence of Theorem 4.10, for if (4.29) did not hold, it would contradict Theorem 4.10.

The next result gives a stronger conclusion than Theorem 4.9, at the expense of only showing a property of the gradient of  $F$  at a randomly selected iterate.

**Corollary 4.11.** *Suppose the conditions of Theorem 4.10 hold. For any  $K \in \mathbb{N}$ , let  $k(K) \in \{1, \dots, K\}$  represent a random index chosen with probabilities proportional to  $\{\alpha_k\}_{k=1}^K$ . Then,  $\|\nabla F(w_{k(K)})\|_2 \xrightarrow{K \rightarrow \infty} 0$  in probability.*

*Proof.* Using Markov's inequality and (4.30a), for any  $\varepsilon > 0$ , we can write

$$\mathbb{P}\{\|\nabla F(w_k)\|_2 \geq \varepsilon\} = \mathbb{P}\{\|\nabla F(w_k)\|_2^2 \geq \varepsilon^2\} \leq \varepsilon^{-2} \mathbb{E}[\mathbb{E}_k[\|\nabla F(w_k)\|_2^2]] \xrightarrow{K \rightarrow \infty} 0,$$

which is the definition of convergence in probability.  $\square$

Finally, we present the following result (with proof in Appendix B) to illustrate that stronger convergence results also follow under additional regularity conditions.

**Corollary 4.12.** *Under the conditions of Theorem 4.10, if we further assume that the objective function  $F$  is twice differentiable, and that the mapping  $w \mapsto \|\nabla F(w)\|_2^2$  has Lipschitz-continuous derivatives, then*

$$\lim_{k \rightarrow \infty} \mathbb{E}[\|\nabla F(w_k)\|_2^2] = 0.$$

## 4.4 Work Complexity for Large-Scale Learning

Our discussion thus far has focused on the convergence properties of SG when minimizing a given objective function representing either expected or empirical risk. However, our investigation would be incomplete without considering how these properties impact the computational workload associated with solving an underlying machine learning problem. As previewed in §3, there are arguments that a more slowly convergent algorithm such as SG, with its sublinear rate of convergence, is more efficient for large-scale learning than (full, batch) gradient-based methods that have a linear rate of convergence. The purpose of this section is to present these arguments in further detail.

As a first attempt for setting up a framework in which one may compare optimization algorithms for large-scale learning, one might be tempted to consider the situation of having a given training set size  $n$  and asking what type of algorithm—e.g., a simple SG or batch gradient method—would provide the best guarantees in terms of achieving a low expected risk. However, such a comparison is difficult to make when one cannot determine the precise trade-off between per-iteration costs and overall progress of the optimization method that one can guarantee.

An easier way to approach the issue is to consider a *big data* scenario with an infinite supply of training examples, but a limited computational time budget. One can then ask whether running a


---

## Page 35

simple optimization algorithm such as SG works better than running a more sophisticated batch optimization algorithm. We follow such an approach now, following the work in [22, 23].

Suppose that both the expected risk  $R$  and the empirical risk  $R_n$  attain their minima with parameter vectors  $w_* \in \arg \min R(w)$  and  $w_n \in \arg \min R_n(w)$ , respectively. In addition, let  $\tilde{w}_n$  be the approximate empirical risk minimizer returned by a given optimization algorithm when the time budget  $\mathcal{T}_{\max}$  is exhausted. The tradeoffs associated with this scenario can be formalized as choosing the family of prediction functions  $\mathcal{H}$ , the number of examples  $n$ , and the optimization accuracy  $\epsilon := \mathbb{E}[R_n(\tilde{w}_n) - R_n(w_n)]$  in order to minimize, within time  $\mathcal{T}_{\max}$ , the *total error*

$$\mathbb{E}[R(\tilde{w}_n)] = \underbrace{R(w_*)}_{\mathcal{E}_{\text{app}}(\mathcal{H})} + \underbrace{\mathbb{E}[R(w_n) - R(w_*)]}_{\mathcal{E}_{\text{est}}(\mathcal{H}, n)} + \underbrace{\mathbb{E}[R(\tilde{w}_n) - R(w_n)]}_{\mathcal{E}_{\text{opt}}(\mathcal{H}, n, \epsilon)}. \quad (4.31)$$

To minimize this error, one needs to balance the contributions from each of the three terms on the right-hand side. For instance, if one decides to make the optimization more accurate—i.e., reducing  $\epsilon$  in the hope of also reducing the *optimization error*  $\mathcal{E}_{\text{opt}}(\mathcal{H}, n, \epsilon)$  (evaluated with respect to  $R$  rather than  $R_n$ )—one might need to make up for the additional computing time by: (i) reducing the sample size  $n$ , potentially increasing the *estimation error*  $\mathcal{E}_{\text{est}}(\mathcal{H}, n)$ ; or (ii) simplifying the function family  $\mathcal{H}$ , potentially increasing the *approximation error*  $\mathcal{E}_{\text{app}}(\mathcal{H})$ .

Useful guidelines for achieving an optimal balance between these errors can be obtained by setting aside the choice of  $\mathcal{H}$  and carrying out a worst-case analysis on the influence of the sample size  $n$  and optimization tolerance  $\epsilon$ , which together only influence the estimation and optimization errors. This simplified set-up can be formalized in terms of the macroscopic optimization problem

$$\min_{n, \epsilon} \mathcal{E}(n, \epsilon) = \mathbb{E}[R(\tilde{w}_n) - R(w_*)] \text{ s.t. } \mathcal{T}(n, \epsilon) \leq \mathcal{T}_{\max}. \quad (4.32)$$

The computing time  $\mathcal{T}(n, \epsilon)$  depends on the details of the optimization algorithm in interesting ways. For example, the computing time of a batch algorithm increases linearly (at least) with the number of examples  $n$ , whereas, crucially, the computing time of a stochastic algorithm is independent of  $n$ . With a batch optimization algorithm, one could consider increasing  $\epsilon$  in order to make time to use more training examples. However, with a stochastic algorithm, one should always use as many examples as possible because the per-iteration computing time is independent of  $n$ .

To be specific, let us compare the solutions of (4.32) for prototypical stochastic and batch methods—namely, simple SG and a batch gradient method—using simplified forms for the worst-case of the error function  $\mathcal{E}$  and the time function  $\mathcal{T}$ . For the error function, a direct application of the uniform laws of large numbers [155] yields

$$\begin{aligned} \mathcal{E}(n, \epsilon) = \mathbb{E}[R(\tilde{w}_n) - R(w_*)] &= \underbrace{\mathbb{E}[R(\tilde{w}_n) - R_n(\tilde{w}_n)]}_{= \mathcal{O}\left(\sqrt{\log(n)/n}\right)} + \underbrace{\mathbb{E}[R_n(\tilde{w}_n) - R_n(w_n)]}_{= \epsilon} \\ &+ \underbrace{\mathbb{E}[R_n(w_n) - R_n(w_*)]}_{\leq 0} + \underbrace{\mathbb{E}[R_n(w_*) - R(w_*)]}_{= \mathcal{O}\left(\sqrt{\log(n)/n}\right)}, \end{aligned}$$

which leads to the upper bound

$$\mathcal{E}(n, \epsilon) = \mathcal{O}\left(\sqrt{\frac{\log(n)}{n}} + \epsilon\right). \quad (4.33)$$


---

## Page 36

The inverse-square-root dependence on the number of examples  $n$  that appears here is typical of statistical problems. However, even faster convergence rates for reducing these terms with respect to  $n$  can be established under specific conditions. For instance, when the loss function is strongly convex [91] or when the data distribution satisfies certain assumptions [154], it is possible to show that

$$\mathcal{E}(n, \epsilon) = \mathcal{O} \left( \frac{\log(n)}{n} + \epsilon \right).$$

To simplify further, let us work with the asymptotic (i.e., for large  $n$ ) equivalence

$$\mathcal{E}(n, \epsilon) \sim \frac{1}{n} + \epsilon, \quad (4.34)$$

which is the fastest rate that remains compatible with elementary statistical results.<sup>6</sup> Under this assumption, noting that the time constraint in (4.32) will always be active (since one can always lower  $\epsilon$ , and hence  $\mathcal{E}(n, \epsilon)$ , by giving more time to the optimization algorithm), and recalling the worst-case computing time bounds introduced in §3.3, one arrives at the following conclusions.

- • A simple SG method can achieve  $\epsilon$ -optimality with a computing time of  $\mathcal{T}_{\text{stoch}}(n, \epsilon) \sim 1/\epsilon$ . Hence, within the time budget  $\mathcal{T}_{\text{max}}$ , the accuracy achieved is proportional to  $1/\mathcal{T}_{\text{max}}$ , regardless of the  $n$ . This means that, to minimize the error  $\mathcal{E}(n, \epsilon)$ , one should simply choose  $n$  as large as possible. Since the maximum number of examples that can be processed by SG during the time budget is proportional to  $\mathcal{T}_{\text{max}}$ , it follows that the optimal error is proportional to  $1/\mathcal{T}_{\text{max}}$ .
- • A batch gradient method can achieve  $\epsilon$ -optimality with a computing time of  $\mathcal{T}_{\text{batch}}(n, \epsilon) \sim n \log(1/\epsilon)$ . This means that, within the time budget  $\mathcal{T}_{\text{max}}$ , it can achieve  $\epsilon$ -optimality by processing  $n \sim \mathcal{T}_{\text{max}}/\log(1/\epsilon)$  examples. One now finds that the optimal error is not necessarily achieved by choosing  $n$  as large as possible, but rather by choosing  $\epsilon$  (which dictates  $n$ ) to minimize (4.34). Differentiating  $\mathcal{E}(n, \epsilon) \sim \log(1/\epsilon)/\mathcal{T}_{\text{max}} + \epsilon$  with respect to  $\epsilon$  and setting the result equal to zero, one finds that optimality is achieved with  $\epsilon \sim 1/\mathcal{T}_{\text{max}}$ , from which it follows that the optimal error is proportional to  $\log(\mathcal{T}_{\text{max}})/\mathcal{T}_{\text{max}} + 1/\mathcal{T}_{\text{max}}$ .

These results are summarized in Table 4.1. Even though a batch approach possesses a better dependency on  $\epsilon$ , this advantage does not make up for its dependence on  $n$ . This is true even though we have employed (4.34), the most favorable rate that one may reasonably assume. In conclusion, we have found that a stochastic optimization algorithm performs better in terms of expected error, and, hence, makes a better learning algorithm in the sense considered here. These observations are supported by practical experience (recall Figure 3.1 in §3.3).

**A Lower Bound** The results reported in Table 4.1 are also notable because the SG algorithm matches a lower complexity bound that has been established for any optimization method employing a *noisy oracle*. To be specific, in the widely employed model for studying optimization algorithms proposed by Nemirovsky and Yudin [106], one assumes that information regarding the objective function is acquired by querying an *oracle*, ignoring the computational demands of doing so. Using

---

<sup>6</sup>For example, suppose that one is estimating the mean of a distribution  $P$  defined on the real line by minimizing the risk  $R(\mu) = \int (x - \mu)^2 dP(x)$ . The convergence rate (4.34) amounts to estimating the distribution mean with a variance proportional to  $1/n$ . A faster convergence rate would violate the Cramer-Rao bound.


---

## Page 37

<table border="1">
<thead>
<tr>
<th></th>
<th>Batch</th>
<th>Stochastic</th>
</tr>
</thead>
<tbody>
<tr>
<td><math>\mathcal{T}(n, \epsilon)</math></td>
<td><math>\sim n \log \left( \frac{1}{\epsilon} \right)</math></td>
<td><math>\frac{1}{\epsilon}</math></td>
</tr>
<tr>
<td><math>\mathcal{E}^*</math></td>
<td><math>\sim \frac{\log(\mathcal{T}_{\max})}{\mathcal{T}_{\max}} + \frac{1}{\mathcal{T}_{\max}}</math></td>
<td><math>\frac{1}{\mathcal{T}_{\max}}</math></td>
</tr>
</tbody>
</table>

Table 4.1: The first row displays the computing times of idealized batch and stochastic optimization algorithms. The second row gives the corresponding solutions of (4.32), assuming (4.34).

such a model, it has been established, e.g., that the full gradient method applied to minimize a strongly convex objective function is not optimal in terms of the accuracy that can be achieved within a given number of calls to the oracle, but that one can achieve an optimal method through acceleration techniques; see §7.2.

The case when only gradient estimates are available through a noisy oracle has been studied, e.g. in [1, 128]. Roughly speaking, these investigations show that, again when minimizing a strongly convex function, no algorithm that performs  $k$  calls to the oracle can guarantee accuracy better than  $\Omega(1/k)$ . As we have seen, SG achieves this lower bound up to constant factors. This analysis applies both for the optimization of expected risk and empirical risk.

## 4.5 Commentary

Although the analysis presented in §4.4 can be quite compelling, it would be premature to conclude that SG is a perfect solution for large-scale machine learning problems. There is, in fact, a large gap between asymptotical behavior and practical realities. Next, we discuss issues related to this gap.

**Fragility of the Asymptotic Performance of SG** The convergence speed given, e.g., by Theorem 4.7, holds when the stepsize constant  $\beta$  exceeds a quantity inversely proportional to the strong convexity parameter  $c$  (see (4.20)). In some settings, determining such a value is relatively easy, such as when the objective function includes a squared  $\ell_2$ -norm regularizer (e.g., as in (2.3)), in which case the regularization parameter provides a lower bound for  $c$ . However, despite the fact that this can work well in practice, it is not completely satisfactory because one should reduce the regularization parameter when the number of samples increases. It is therefore desirable to design algorithms that adapt to local convexity properties of the objective, so as to avoid having to place cumbersome restrictions on the stepsizes.

**SG and Ill-Conditioning** The analysis of §4.4 is compelling since, as long as the optimization problem is reasonably well-conditioned, the constant factors favor the SG algorithm. In particular, the minimalism of the SG algorithm allows for very efficient implementations that either fully leverage the sparsity of training examples (as in the case study on text classification in §2.1) or harness the computational power of GPU processors (as in the case study on deep neural network in §2.2). In contrast, state-of-the-art batch optimization algorithms often carry more overhead. However, this advantage erodes when the conditioning of the objective function worsens. Again, consider Theorem 4.7. This result involves constant factors that grow with both the condition


---

## Page 38

number  $L/c$  and the ratio  $M/c$ . Both of these ratios can be improved greatly by adaptively rescaling the stochastic directions based on matrices that capture local curvature information of the objective function; see [6](#).

**Opportunities for Distributed Computing** Distributing the SG step computation can potentially reduce the computing time by a constant factor equal to the number of machines. However, such an improvement is difficult to realize. The SG algorithm is notoriously difficult to distribute efficiently because it accesses the shared parameter vector  $w$  with relatively high frequency. Consequently, even though it is very robust to additional noise and can be run with very relaxed synchronization [[112](#), [45](#)], distributed SG algorithms suffer from large communication overhead. Since this overhead is potentially much larger than the additional work associated with mini-batch and other methods with higher per-iteration costs, distributed computing offers new opportunities for the success of such methods for machine learning.

**Alternatives with Faster Convergence** As mentioned above, [[1](#), [128](#)] establish lower complexity bounds for optimization algorithms that only access information about the objective function through noisy estimates of  $F(w_k)$  and  $\nabla F(w_k)$  acquired in each iteration. The bounds apply, e.g., when SG is employed to minimize the expected risk  $R$  using gradient estimates evaluated on samples drawn from the distribution  $P$ . However, an algorithm that optimizes the empirical risk  $R_n$  has access to an additional piece of information: it knows when a gradient estimate is evaluated on a training example that has already been visited during previous iterations. Recent *gradient aggregation* methods (see [§5.3](#)) make use of this information and improve upon the lower bound in [[1](#)] for the optimization of the empirical risk (though not for the expected risk). These algorithms enjoy linear convergence with low computing times in practice. Another avenue for improving the convergence rate is to employ a *dynamic sampling* approach (see [§5.2](#)), which, as we shall see, can match the optimal asymptotic efficiency of SG in big data settings.


---

## Page 39

### Inset 4.3: Regret Bounds

Convergence results for SG and its variants are occasionally established using *regret bounds* as an intermediate step [164, 144, 54]. Regret bounds can be traced to Novikoff’s analysis of the Perceptron [115] and to Cover’s universal portfolios [41]. To illustrate, suppose that one aims to minimize a convex expected risk measure  $R(w) = \mathbb{E}[f(w; \xi)]$  over  $w \in \mathbb{R}^d$  with minimizer  $w_* \in \mathbb{R}^d$ . At a given iterate  $w_k$ , one obtains by convexity of  $f(w; \xi_k)$  (recall (A.1)) that

$$\|w_{k+1} - w_*\|^2 - \|w_k - w_*\|^2 \leq -2\alpha_k(f(w_k; \xi_k) - f(w_*; \xi_k)) + \alpha_k^2 \|\nabla f(w_k; \xi_k)\|_2^2.$$

Following [164], assuming that  $\|\nabla f(w_k; \xi_k)\|_2^2 \leq M$  and  $\|w_k - w_*\|_2^2 < B$  for some constants  $M > 0$  and  $B > 0$  for all  $k \in \mathbb{N}$ , one finds that

$$\begin{aligned} & \alpha_{k+1}^{-1} \|w_{k+1} - w_*\|_2^2 - \alpha_k^{-1} \|w_k - w_*\|_2^2 \\ & \leq -2(f(w_k; \xi_k) - f(w_*; \xi_k)) + \alpha_k M + (\alpha_{k+1}^{-1} - \alpha_k^{-1}) \|w_k - w_*\|_2^2 \\ & \leq -2(f(w_k; \xi_k) - f(w_*; \xi_k)) + \alpha_k M + (\alpha_{k+1}^{-1} - \alpha_k^{-1}) B. \end{aligned}$$

Summing for  $k = \{1 \dots K\}$  with stepsizes  $\alpha_k = 1/\sqrt{k}$  leads to the regret bound

$$\left( \sum_{k=1}^K f(w_k; \xi_k) \right) \leq \left( \sum_{k=1}^K f(w_*; \xi_k) \right) + M\sqrt{K} + o(\sqrt{K}), \quad (4.35)$$

which bounds the losses incurred from  $\{w_k\}$  compared to those yielded by the fixed vector  $w_*$ . Such a bound is remarkable because its derivation holds for any sequence of noise variables  $\{\xi_k\}$ . This means that the average loss observed during the execution of the SG algorithm is *never* much worse than the best average loss one would have obtained if the optimal parameter  $w_*$  were known in advance. Further assuming that the noise variables are independent and using a martingale argument [34] leads to more traditional results of the form

$$\mathbb{E} \left[ \frac{1}{K} \sum_{k=1}^K F(w_k) \right] \leq F_* + \mathcal{O} \left( \frac{1}{\sqrt{K}} \right).$$

As long as one makes the same independent noise assumption, results obtained with this technique cannot be fundamentally different from the results that we have established. However, one should appreciate that the regret bound (4.35) itself remains meaningful when the noise variables are dependent or even adversarial. Such results reveals interesting connections between probability theory, machine learning, and game theory [143, 34].


---

## Page 40

## 5 Noise Reduction Methods

The theoretical arguments in the previous section, together with extensive computational experience, have led many in the machine learning community to view SG as the ideal optimization approach for large-scale applications. We argue, however, that this is far from settled. SG suffers from, amongst other things, the adverse effect of noisy gradient estimates. This prevents it from converging to the solution when fixed stepsizes are used and leads to a slow, sublinear rate of convergence when a diminishing stepsize sequence  $\{\alpha_k\}$  is employed.

To address this limitation, methods endowed with *noise reduction* capabilities have been developed. These methods, which reduce the errors in the gradient estimates and/or iterate sequence, have proved to be effective in practice and enjoy attractive theoretical properties. Recalling the schematic of optimization methods in Figure 3.3, we depict these methods on the horizontal axis given in Figure 5.1.

The diagram illustrates the spectrum of optimization methods. A horizontal arrow represents the progression from 'Stochastic gradient method' to 'Batch gradient method'. A red dot is located on this arrow. Below the arrow, a list of noise reduction methods is shown: 'Dynamic sampling', 'Gradient aggregation', and 'Iterate averaging'. A dashed blue arrow points from the 'Stochastic gradient method' down to the 'Stochastic Newton method'.

Fig. 5.1: View of the schematic from Figure 3.3 with a focus on noise reduction methods.

The first two classes of methods that we consider achieve noise reduction in a manner that allows them to possess a linear rate of convergence to the optimal value using a fixed stepsize. The first type, *dynamic sampling* methods, achieve noise reduction by gradually increasing the mini-batch size used in the gradient computation, thus employing increasingly more accurate gradient estimates as the optimization process proceeds. *Gradient aggregation* methods, on the other hand, improve the quality of the search directions by storing gradient estimates corresponding to samples employed in previous iterations, updating one (or some) of these estimates in each iteration, and defining the search direction as a weighted average of these estimates.

The third class of methods that we consider, *iterate averaging* methods, accomplish noise reduction not by averaging gradient estimates, but by maintaining an average of iterates computed during the optimization process. Employing a more aggressive stepsize sequence—of order  $\mathcal{O}(1/\sqrt{k})$  rather than  $\mathcal{O}(1/k)$ , which is appealing in itself—it is this sequence of averaged iterates that converges to the solution. These methods are closer in spirit to SG and their rate of convergence remains sublinear, but it can be shown that the variance of the sequence of average iterates is smaller than the variance of the SG iterates.

To formally motivate a concept of noise reduction, we begin this section by discussing a fundamental result that stipulates a rate of decrease in noise that allows a stochastic-gradient-type


---

## Page 41

method to converge at a linear rate. We then show that a dynamic sampling method that enforces such noise reduction enjoys optimal complexity bounds, as defined in §4.4. Next, we discuss three gradient aggregation methods—SVRG, SAGA, and SAG—the first of which can be viewed as a bridge between methods that control errors in the gradient with methods like SAGA and SAG in which noise reduction is accomplished in a more subtle manner. We conclude with a discussion of the practical and theoretical properties of iterate averaging methods.

## 5.1 Reducing Noise at a Geometric Rate

Let us recall the fundamental inequality (4.4), which we restate here for convenience:

$$\mathbb{E}_{\xi_k}[F(w_{k+1})] - F(w_k) \leq -\alpha_k \nabla F(w_k)^T \mathbb{E}_{\xi_k}[g(w_k, \xi_k)] + \frac{1}{2} \alpha_k^2 L \mathbb{E}_{\xi_k}[\|g(w_k, \xi_k)\|_2^2].$$

(Recall that, as stated in (4.1), the objective  $F$  could stand for the expected risk  $R$  or empirical risk  $R_n$ .) It is intuitively clear that if  $-g(w_k, \xi_k)$  is a descent direction in expectation (making the first term on the right hand side negative) and if we are able to decrease  $\mathbb{E}_{\xi_k}[\|g(w_k, \xi_k)\|_2^2]$  fast enough (along with  $\|\nabla F(w_k)\|_2^2$ ), then the effect of having noisy directions will not impede a fast rate of convergence. From another point of view, we can expect such behavior if, in Assumption 4.3, we suppose instead that the variance of  $g(w_k, \xi_k)$  vanishes sufficiently quickly.

We formalize this intuitive argument by considering the SG method with a fixed stepsize and showing that the sequence of expected optimality gaps vanishes at a linear rate as long as the variance of the stochastic vectors, denoted by  $\mathbb{V}_{\xi_k}[g(w_k, \xi_k)]$  (recall (4.6)), decreases *geometrically*.

**Theorem 5.1 (Strongly Convex Objective, Noise Reduction).** *Suppose that Assumptions 4.1, 4.3, and 4.5 (with  $F_{\inf} = F_*$ ) hold, but with (4.8) refined to the existence of constants  $M \geq 0$  and  $\zeta \in (0, 1)$  such that, for all  $k \in \mathbb{N}$ ,*

$$\mathbb{V}_{\xi_k}[g(w_k, \xi_k)] \leq M \zeta^{k-1}. \quad (5.1)$$

*In addition, suppose that the SG method (Algorithm 4.1) is run with a fixed stepsize,  $\alpha_k = \bar{\alpha}$  for all  $k \in \mathbb{N}$ , satisfying*

$$0 < \bar{\alpha} \leq \min \left\{ \frac{\mu}{L\mu_G^2}, \frac{1}{c\mu} \right\}. \quad (5.2)$$

*Then, for all  $k \in \mathbb{N}$ , the expected optimality gap satisfies*

$$\mathbb{E}[F(w_k) - F_*] \leq \omega \rho^{k-1}, \quad (5.3)$$

*where*

$$\omega := \max \left\{ \frac{\bar{\alpha}LM}{c\mu}, F(w_1) - F_* \right\} \quad (5.4a)$$

$$\text{and } \rho := \max \left\{ 1 - \frac{\bar{\alpha}c\mu}{2}, \zeta \right\} < 1. \quad (5.4b)$$

*Proof.* By Lemma 4.4 (specifically, (4.10a)), we have

$$\mathbb{E}_{\xi_k}[F(w_{k+1})] - F(w_k) \leq -\mu \bar{\alpha} \|\nabla F(w_k)\|_2^2 + \frac{1}{2} \bar{\alpha}^2 L \mathbb{E}_{\xi_k}[\|g(w_k, \xi_k)\|_2^2].$$


---

## Page 42

Hence, from (4.6), (4.7b), (5.1), (5.2), and (4.12), we have

$$\begin{aligned}
\mathbb{E}_{\xi_k}[F(w_{k+1})] - F(w_k) &\leq -\mu\bar{\alpha}\|\nabla F(w_k)\|_2^2 + \frac{1}{2}\bar{\alpha}^2 L \left( \mu_G^2 \|\nabla F(w_k)\|_2^2 + M\zeta^{k-1} \right) \\
&\leq -\left(\mu - \frac{1}{2}\bar{\alpha}L\mu_G^2\right)\bar{\alpha}\|\nabla F(w_k)\|_2^2 + \frac{1}{2}\bar{\alpha}^2 LM\zeta^{k-1} \\
&\leq -\frac{1}{2}\mu\bar{\alpha}\|\nabla F(w_k)\|_2^2 + \frac{1}{2}\bar{\alpha}^2 LM\zeta^{k-1} \\
&\leq -\bar{\alpha}c\mu(F(w_k) - F_*) + \frac{1}{2}\bar{\alpha}^2 LM\zeta^{k-1}.
\end{aligned}$$

Adding and subtracting  $F_*$  and taking total expectations, this yields

$$\mathbb{E}[F(w_{k+1}) - F_*] \leq (1 - \bar{\alpha}c\mu)\mathbb{E}[F(w_k) - F_*] + \frac{1}{2}\bar{\alpha}^2 LM\zeta^{k-1}. \quad (5.5)$$

We now use induction to prove (5.3). By the definition of  $\omega$ , it holds for  $k = 1$ . Then, assuming it holds for  $k \geq 1$ , we have from (5.5), (5.4a), and (5.4b) that

$$\begin{aligned}
\mathbb{E}[F(w_{k+1}) - F_*] &\leq (1 - \bar{\alpha}c\mu)\omega\rho^{k-1} + \frac{1}{2}\bar{\alpha}^2 LM\zeta^{k-1} \\
&= \omega\rho^{k-1} \left( 1 - \bar{\alpha}c\mu + \frac{\bar{\alpha}^2 LM}{2\omega} \left( \frac{\zeta}{\rho} \right)^{k-1} \right) \\
&\leq \omega\rho^{k-1} \left( 1 - \bar{\alpha}c\mu + \frac{\bar{\alpha}^2 LM}{2\omega} \right) \\
&\leq \omega\rho^{k-1} \left( 1 - \bar{\alpha}c\mu + \frac{\bar{\alpha}c\mu}{2} \right) \\
&= \omega\rho^{k-1} \left( 1 - \frac{\bar{\alpha}c\mu}{2} \right) \\
&\leq \omega\rho^k,
\end{aligned}$$

as desired.  $\square$

Consideration of the typical magnitudes of the constants  $\mu$ ,  $L$ ,  $\mu_G$ , and  $c$  in (5.2) reveals that the admissible range of values of  $\bar{\alpha}$  is large, i.e., the restriction on the stepsize  $\bar{\alpha}$  is not unrealistic in practical situations.

Now, a natural question is whether one can design efficient optimization methods for attaining the critical bound (5.1) on the variance of the stochastic directions. We show next that a dynamic sampling method is one such approach.

## 5.2 Dynamic Sample Size Methods

Consider the iteration

$$w_{k+1} \leftarrow w_k - \bar{\alpha}g(w_k, \xi_k), \quad (5.6)$$

where the stochastic directions are computed for some  $\tau > 1$  as

$$g(w_k, \xi_k) := \frac{1}{n_k} \sum_{i \in \mathcal{S}_k} \nabla f(w_k; \xi_{k,i}) \quad \text{with} \quad n_k := |\mathcal{S}_k| = \lceil \tau^{k-1} \rceil. \quad (5.7)$$

That is, consider a mini-batch SG iteration with a fixed stepsize in which the mini-batch size used to compute unbiased stochastic gradient estimates increases geometrically as a function of the


---

## Page 43

iteration counter  $k$ . To show that such an approach can fall under the class of methods considered in Theorem 5.1, suppose that the samples represented by the random variables  $\{\xi_{k,i}\}_{i \in \mathcal{S}_k}$  are drawn independently according to  $P$  for all  $k \in \mathbb{N}$ . If we assume that each stochastic gradient  $\nabla f(w_k; \xi_{k,i})$  has an expectation equal to the true gradient  $\nabla F(w_k)$ , then (4.7a) holds with  $\mu_G = \mu = 1$ . If, in addition, the variance of each such stochastic gradient is equal and is bounded by  $M \geq 0$ , then for arbitrary  $i \in \mathcal{S}_k$  we have (see [61, p.183])

$$\mathbb{V}_{\xi_k}[g(w_k, \xi_k)] \leq \frac{\mathbb{V}_{\xi_k}[\nabla f(w_k; \xi_{k,i})]}{n_k} \leq \frac{M}{n_k}. \quad (5.8)$$

This bound, when combined with the rate of increase in  $n_k$  given in (5.7), yields (5.1). We have thus shown that if one employs a mini-batch SG method with (unbiased) gradient estimates computed as in (5.7), then, by Theorem 5.1, one obtains linear convergence to the optimal value of a strongly convex function. We state this formally as the following corollary to Theorem 5.1.

**Corollary 5.2.** *Let  $\{w_k\}$  be the iterates generated by (5.6)–(5.7) with unbiased gradient estimates, i.e.,  $\mathbb{E}_{\xi_{k,i}}[\nabla f(w_k; \xi_{k,i})] = \nabla F(w_k)$  for all  $k \in \mathbb{N}$  and  $i \in \mathcal{S}_k$ . Then, the variance condition (5.1) is satisfied, and if all other assumptions of Theorem 5.1 hold, then the expected optimality gap vanishes linearly in the sense of (5.3).*

The reader may question whether it is meaningful to describe a method as linearly convergent if the per-iteration cost increases without bound. In other words, it is not immediately apparent that such an algorithm is competitive with a classical SG approach even though the desired reduction in the gradient variance is achieved. To address this question, let us estimate the total work complexity of the dynamic sampling algorithm, defined as the number of evaluations of the individual gradients  $\nabla f(w_k; \xi_{k,i})$  required to compute an  $\epsilon$ -optimal solution, i.e., to achieve

$$\mathbb{E}[F(w_k) - F_*] \leq \epsilon. \quad (5.9)$$

We have seen that the simple SG method (3.7) requires one such evaluation per iteration, and that its rate of convergence for diminishing stepsizes (i.e., the only set-up in which convergence to the solution can be guaranteed) is given by (4.21). Therefore, as previously mentioned, the number of stochastic gradient evaluations required by the SG method to guarantee (5.9) is  $\mathcal{O}(\epsilon^{-1})$ . We now show that the method (5.6)–(5.7) can attain the same complexity.

**Theorem 5.3.** *Suppose that the dynamic sampling SG method (5.6)–(5.7) is run with a stepsize  $\bar{\alpha}$  satisfying (5.2) and some  $\tau \in (1, (1 - \frac{\bar{\alpha}c\mu}{2})^{-1}]$ . In addition, suppose that Assumptions 4.1, 4.3, and Assumption 4.5 (with  $F_{\inf} = F_*$ ) hold. Then, the total number of evaluations of a stochastic gradient of the form  $\nabla f(w_k; \xi_{k,i})$  required to obtain (5.9) is  $\mathcal{O}(\epsilon^{-1})$ .*

*Proof.* We have that the conditions of Theorem 5.1 hold with  $\zeta = 1/\tau$ . Hence, we have from (5.3) that there exists  $\bar{k} \in \mathbb{N}$  such that (5.9) holds for all  $k \geq \bar{k}$ . We can then use  $\omega\rho^{\bar{k}-1} \leq \epsilon$  to write  $(\bar{k} - 1) \log \rho \leq \log(\epsilon/\omega)$ , which along with  $\rho \in (0, 1)$  (recall (5.4b)) implies that

$$\bar{k} - 1 \geq \left\lceil \frac{\log(\epsilon/\omega)}{\log \rho} \right\rceil = \left\lceil \frac{\log(\omega/\epsilon)}{-\log \rho} \right\rceil. \quad (5.10)$$

Let us now estimate the total number of sample gradient evaluations required up through iteration  $\bar{k}$ . We claim that, without loss of generality, we may assume that  $\log(\omega/\epsilon)/(-\log \rho)$  is


---

## Page 44

integer-valued and that (5.10) holds at equality. Then, by (5.7), the number of sample gradients required in iteration  $\bar{k}$  is  $\lceil \tau^{\bar{k}-1} \rceil$  where

$$\begin{aligned}
\tau^{\bar{k}-1} &= \tau^{\frac{\log(\omega/\epsilon)}{-\log \rho}} \\
&= \exp \left( \log \left( \tau^{\frac{\log(\omega/\epsilon)}{-\log \rho}} \right) \right) \\
&= \exp \left( \left( \frac{\log(\omega/\epsilon)}{-\log \rho} \right) \log \tau \right) \\
&= (\exp(\log(\omega/\epsilon)))^{\frac{\log \tau}{-\log \rho}} \\
&= \left( \frac{\omega}{\epsilon} \right)^\theta \quad \text{with } \theta := \frac{\log \tau}{-\log \rho}.
\end{aligned}$$

Therefore, the total number of sample gradient evaluations for the first  $\bar{k}$  iterations is

$$\sum_{j=1}^{\bar{k}} \lceil \tau^{j-1} \rceil \leq 2 \sum_{j=1}^{\bar{k}} \tau^{j-1} = 2 \left( \frac{\tau^{\bar{k}} - \tau}{\tau - 1} \right) = 2 \left( \frac{\tau(\omega/\epsilon)^\theta - \tau}{\tau - 1} \right) \leq 2 \left( \frac{\omega}{\epsilon} \right)^\theta \left( \frac{1}{1 - 1/\tau} \right).$$

In fact, since  $\tau \leq (1 - \frac{\bar{\alpha}c\mu}{2})^{-1}$ , it follows from (5.4b) that  $\rho = \zeta = \tau^{-1}$ , which implies that  $\theta = 1$ . Specifically, with  $\tau = (1 - \sigma(\frac{\bar{\alpha}c\mu}{2}))^{-1}$  for some  $\sigma \in [0, 1]$ , then  $\theta = 1$  and

$$\sum_{j=1}^{\bar{k}} \lceil \tau^{j-1} \rceil \leq \frac{4\omega}{\sigma\epsilon\bar{\alpha}c\mu},$$

as desired.  $\square$

The discussion so far has focused on dynamic sampling strategies for a gradient method, but these ideas also apply for second-order methods that incorporate the matrix  $H$  as in (4.2).

This leads to the following question: given the rate of convergence of a batch optimization algorithm on strongly convex functions (i.e., linear, superlinear, etc.), what should be the sampling rate so that the overall algorithm is *efficient* in the sense that it results in the lowest computational complexity? To answer this question, certain results have been established [120]: (i) if the optimization method has a sublinear rate of convergence, then there is no sampling rate that makes the algorithm “efficient”; (ii) if the optimization algorithm is linearly convergent, then the sampling rate must be geometric (with restrictions on the constant in the rate) for the algorithm to be “efficient”; (iii) for superlinearly convergent methods, increasing the sample size at a rate that is slightly faster than geometric will yield an “efficient” method.

### 5.2.1 Practical Implementation

We now address the question of how to design practical algorithms that enjoy the theoretical benefits of noise reduction through the use of increasing sample sizes.

One approach is to follow the theoretical results described above and tie the rate of growth in the sample size  $n_k = |\mathcal{S}_k|$  to the rate of convergence of the optimization iteration [62, 77]. Such an approach, which *presents* the sampling rate before running the optimization algorithm, requires some


---

## Page 45

experimentation. For example, for the iteration (5.6), one needs to find a value of the parameter  $\tau > 1$  in (5.7) that yields good performance for the application at hand. In addition, one may want to delay the application of dynamic sampling to prevent the full sample set from being employed too soon (or at all). Such heuristic adaptations could be difficult in practice.

An alternative is to consider mechanisms for choosing the sample sizes not according to a prescribed sequence, but *adaptively* according to information gathered during the optimization process. One avenue that has been explored along these lines has been to design techniques that produce descent directions sufficiently often [28, 73]. Such an idea is based on two observations. First, one can show that *any* direction  $g(w_k, \xi_k)$  is a descent direction for  $F$  at  $w_k$  if, for some  $\chi \in [0, 1)$ , one has

$$\delta(w_k, \xi_k) := \|g(w_k, \xi_k) - \nabla F(w_k)\|_2 \leq \chi \|g(w_k, \xi_k)\|_2. \quad (5.11)$$

To see this, note that  $\|\nabla F(w_k)\|_2 \geq (1 - \chi) \|g(w_k, \xi_k)\|_2$ , which after squaring (5.11) implies

$$\begin{aligned} \nabla F(w_k)^T g(w_k, \xi_k) &\geq \frac{1}{2} (1 - \chi^2) \|g(w_k, \xi_k)\|_2^2 + \|\nabla F(w_k)\|_2^2 \\ &\geq \frac{1}{2} (1 - \chi^2 + (1 - \chi)^2) \|g(w_k, \xi_k)\|_2^2 \\ &\geq (1 - \chi) \|g(w_k, \xi_k)\|_2^2. \end{aligned}$$

The second observation is that while one cannot cheaply verify the inequality in (5.11) because it involves the evaluation of  $\nabla F(w_k)$ , one can estimate the left-hand side  $\delta(w_k, \xi_k)$ , and then choose  $n_k$  so that (5.11) holds sufficiently often.

Specifically, if we assume that  $g(w_k, \xi_k)$  is an unbiased gradient estimate, then

$$\mathbb{E}[\delta(w_k, \xi_k)^2] = \mathbb{E}[\|g(w_k, \xi_k) - \nabla F(w_k)\|_2^2] = \mathbb{V}_{\xi_k}[g(w_k, \xi_k)].$$

This variance is expensive to compute, but one can approximate it with a sample variance. For example, if the samples are drawn without replacement from a set of (very large) size  $n_k$ , then one has the approximation

$$\mathbb{V}_{\xi_k}[g(w_k, \xi_k)] \approx \frac{\text{trace}(\text{Cov}(\{\nabla f(w_k; \xi_{k,i})\}_{i \in \mathcal{S}_k}))}{n_k} =: \varphi_k.$$

An *adaptive sampling* algorithm thus tests the following condition in place of (5.11):

$$\varphi_k \leq \chi^2 \|g(w_k, \xi_k)\|_2^2. \quad (5.12)$$

If this condition is not satisfied, then one may consider increasing the sample size—either immediately in iteration  $k$  or in a subsequent iteration—to a size that one might predict would satisfy such a condition. This technique is algorithmically attractive, but does not guarantee that the sample size  $n_k$  increases at a geometric rate. One can, however, employ a backup [28, 73]: if (5.12) increases the sampling rate more slowly than a preset geometric sequence, then a growth in the sample size is imposed.

Dynamic sampling techniques are not yet widely used in machine learning, and we suspect that the practical technique presented here might merely serve as a starting point for further investigations. Ultimately, an algorithm that performs like SG at the start and transitions to a regime of reduced variance in an efficient manner could prove to be a very powerful method for large-scale learning.


---

## Page 46

### 5.3 Gradient Aggregation

The dynamic sample size methods described in the previous subsection require a careful balance in order to achieve the desired linear rate of convergence without jeopardizing per-iteration costs. Alternatively, one can attempt to achieve an improved rate by asking a different question: rather than compute increasingly more *new* stochastic gradient information in each iteration, is it possible to achieve a lower variance by *reusing* and/or *revising* previously computed information? After all, if the current iterate has not been displaced too far from previous iterates, then stochastic gradient information from previous iterates may still be useful. In addition, if one maintains indexed gradient estimates in storage, then one can revise specific estimates as new information is collected. Ideas such as these lead to concepts of *gradient aggregation*. In this subsection, we present ideas for gradient aggregation in the context of minimizing a finite sum such as an empirical risk measure  $R_n$ , for which they were originally designed.

Gradient aggregation algorithms for minimizing finite sums that possess cheap per-iteration costs have a long history. For example, Bertsekas and co-authors [13] have proposed *incremental gradient* methods, the randomized versions of which can be viewed as instances of a basic SG method for minimizing a finite sum. However, the basic variants of these methods only achieve a sublinear rate of convergence. By contrast, the methods on which we focus in this section are able to achieve a linear rate of convergence on strongly convex problems. This improved rate is achieved primarily by either an increase in computation or an increase in storage.

#### 5.3.1 SVRG

The first method we consider operates in cycles. At the beginning of each cycle, an iterate  $w_k$  is available at which the algorithm computes a batch gradient  $\nabla R_n(w_k) = \frac{1}{n} \sum_{i=1}^n \nabla f_i(w_k)$ . Then, after initializing  $\tilde{w}_1 \leftarrow w_k$ , a set of  $m$  inner iterations indexed by  $j$  with an update  $\tilde{w}_{j+1} \leftarrow \tilde{w}_j - \alpha \tilde{g}_j$  are performed, where

$$\tilde{g}_j \leftarrow \nabla f_{i_j}(\tilde{w}_j) - (\nabla f_{i_j}(w_k) - \nabla R_n(w_k)) \quad (5.13)$$

and  $i_j \in \{1, \dots, n\}$  is chosen at random. This formula has a simple interpretation. Since the expected value of  $\nabla f_{i_j}(w_k)$  over all possible  $i_j \in \{1, \dots, n\}$  is equal to  $\nabla R_n(w_k)$ , one can view  $\nabla f_{i_j}(w_k) - \nabla R_n(w_k)$  as the bias in the gradient estimate  $\nabla f_{i_j}(w_k)$ . Thus, in every iteration, the algorithm randomly draws a stochastic gradient  $\nabla f_{i_j}(\tilde{w}_j)$  evaluated at the current inner iterate  $\tilde{w}_j$  and corrects it based on a perceived bias. Overall,  $\tilde{g}_j$  represents an unbiased estimator of  $\nabla R_n(\tilde{w}_j)$ , but with a variance that one can expect to be smaller than if one were simply to chose  $\tilde{g}_j = \nabla f_{i_j}(\tilde{w}_j)$  (as in simple SG). This is the reason that the method is referred to as the *stochastic variance reduced gradient* (SVRG) method.

A formal description of a few variants of SVRG is presented as Algorithm 5.1. For both options (b) and (c), it has been shown that when applied to minimize a strongly convex  $R_n$ , Algorithm 5.1 can achieve a linear rate of convergence [82]. More precisely, if the stepsize  $\alpha$  and the length of the inner cycle  $m$  are chosen so that

$$\rho := \frac{1}{1 - 2\alpha L} \left( \frac{1}{m\alpha} + 2L\alpha \right) < 1,$$

then, given that the algorithm has reached  $w_k$ , one obtains

$$\mathbb{E}[R_n(w_{k+1}) - R_n(w_*)] \leq \rho \mathbb{E}[R_n(w_k) - R_n(w_*)]$$


---

## Page 47

(where expectation is taken with respect to the random variables in the inner cycle). It should be emphasized that resulting linear convergence rate applies to the outer iterates  $\{w_k\}$ , where each step from  $w_k$  to  $w_{k+1}$  requires  $2m + n$  evaluations of component gradients: Step 7 requires two stochastic gradient evaluations while Step 3 requires  $n$  (a full gradient). Therefore, one iteration of SVRG is much more expensive than one in SG, and in fact is comparable to a full gradient iteration.

---

**Algorithm 5.1** SVRG Methods for Minimizing an Empirical Risk  $R_n$

---

```

1: Choose an initial iterate  $w_1 \in \mathbb{R}^d$ , stepsize  $\alpha > 0$ , and positive integer  $m$ .
2: for  $k = 1, 2, \dots$  do
3:   Compute the batch gradient  $\nabla R_n(w_k)$ .
4:   Initialize  $\tilde{w}_1 \leftarrow w_k$ .
5:   for  $j = 1, \dots, m$  do
6:     Choose  $i_j$  uniformly from  $\{1, \dots, n\}$ .
7:     Set  $\tilde{g}_j \leftarrow \nabla f_{i_j}(\tilde{w}_j) - (\nabla f_{i_j}(w_k) - \nabla R_n(w_k))$ .
8:     Set  $\tilde{w}_{j+1} \leftarrow \tilde{w}_j - \alpha \tilde{g}_j$ .
9:   end for
10:  Option (a): Set  $w_{k+1} = \tilde{w}_{m+1}$ 
11:  Option (b): Set  $w_{k+1} = \frac{1}{m} \sum_{j=1}^m \tilde{w}_{j+1}$ 
12:  Option (c): Choose  $j$  uniformly from  $\{1, \dots, m\}$  and set  $w_{k+1} = \tilde{w}_{j+1}$ .
13: end for

```

---

In practice, SVRG appears to be quite effective in certain applications compared with SG if one requires high training accuracy. For the first epochs, SG is more efficient, but once the iterates approach the solution the benefits of the fast convergence rate of SVRG can be observed. Without explicit knowledge of  $L$  and  $c$ , the length of the inner cycle  $m$  and the stepsize  $\alpha$  are typically both chosen by experimentation.

### 5.3.2 SAGA

The second method we consider employs an iteration that is closer in form to SG in that it does not operate in cycles, nor does it compute batch gradients (except possibly at the initial point). Instead, in each iteration, it computes a stochastic vector  $g_k$  as the average of stochastic gradients evaluated at previous iterates. Specifically, in iteration  $k$ , the method will have stored  $\nabla f_i(w_{[i]})$  for all  $i \in \{1, \dots, n\}$  where  $w_{[i]}$  represents the latest iterate at which  $\nabla f_i$  was evaluated. An integer  $j \in \{1, \dots, n\}$  is then chosen at random and the stochastic vector is set by

$$g_k \leftarrow \nabla f_j(w_k) - \nabla f_j(w_{[j]}) + \frac{1}{n} \sum_{i=1}^n \nabla f_i(w_{[i]}). \quad (5.14)$$

Taking the expectation of  $g_k$  with respect to all choices of  $j \in \{1, \dots, n\}$ , one again has that  $\mathbb{E}[g_k] = \nabla R_n(w_k)$ . Thus, the method employs unbiased gradient estimates, but with variances that are expected to be less than the stochastic gradients that would be employed in a basic SG routine. A precise algorithm employing (5.14), referred to as SAGA [46], is given in Algorithm 5.2.

Beyond its initialization phase, the per-iteration cost of Algorithm 5.2 is the same as in a basic SG method. However, it has been shown that the method can achieve a linear rate of convergence


---

## Page 48

---

**Algorithm 5.2** SAGA Method for Minimizing an Empirical Risk  $R_n$ 


---

```

1: Choose an initial iterate  $w_1 \in \mathbb{R}^d$  and stepsize  $\alpha > 0$ .
2: for  $i = 1, \dots, n$  do
3:   Compute  $\nabla f_i(w_1)$ .
4:   Store  $\nabla f_i(w_{[i]}) \leftarrow \nabla f_i(w_1)$ .
5: end for
6: for  $k = 1, 2, \dots$  do
7:   Choose  $j$  uniformly in  $\{1, \dots, n\}$ .
8:   Compute  $\nabla f_j(w_k)$ .
9:   Set  $g_k \leftarrow \nabla f_j(w_k) - \nabla f_j(w_{[j]}) + \frac{1}{n} \sum_{i=1}^n \nabla f_i(w_{[i]})$ .
10:  Store  $\nabla f_j(w_{[j]}) \leftarrow \nabla f_j(w_k)$ .
11:  Set  $w_{k+1} \leftarrow w_k - \alpha g_k$ .
12: end for

```

---

when minimizing a strongly convex  $R_n$ . Specifically, with  $\alpha = 1/(2(cn + L))$ , one can show that

$$\mathbb{E}[\|w_k - w_*\|_2^2] \leq \left(1 - \frac{c}{2(cn + L)}\right)^k \left(\|w_1 - w_*\|_2^2 + \frac{nD}{cn + L}\right),$$

where  $D := R_n(w_1) - \nabla R_n(w_*)^T(w_1 - w_*) - R_n(w_*)$ .

Of course, attaining such a result requires knowledge of the strong convexity constant  $c$  and Lipschitz constant  $L$ . If  $c$  is not known, then the stepsize can instead be chosen to be  $\alpha = 1/(3L)$  and a similar convergence result can be established; see [46].

Alternative initialization techniques could be used in practice, which may be more effective than evaluating all the gradients  $\{\nabla f_i\}_{i=1}^n$  at the initial point. For example, one could perform one epoch of simple SG, or one can assimilate iterates one-by-one and compute  $g_k$  only using the gradients available up to that point.

One important drawback of Algorithm 5.2 is the need to store  $n$  stochastic gradient vectors, which would be prohibitive in many large-scale applications. Note, however, that if the component functions are of the form  $f_i(w_k) = \hat{f}(x_i^T w_k)$ , then

$$\nabla f_i(w_k) = \hat{f}'(x_i^T w_k) x_i.$$

That is, when the feature vectors  $\{x_i\}$  are already available in storage, one need only store the scalar  $\hat{f}'(x_i^T w_k)$  in order to construct  $\nabla f_i(w_k)$  at a later iteration. Such a functional form of  $f_i$  occurs in logistic and least squares regression.

Algorithm 5.2 has its origins in the *stochastic average gradient* (SAG) algorithm [139, 90], where the stochastic direction is defined as

$$g_k \leftarrow \frac{1}{n} \left( \nabla f_j(w_k) - \nabla f_j(w_{[j]}) + \sum_{i=1}^n \nabla f_i(w_{[i]}) \right). \quad (5.15)$$

Although this  $g_k$  is not an unbiased estimator of  $\nabla R_n(w_k)$ , the method enjoys a linear rate of convergence. One finds that the SAG algorithm is a randomized version of the Incremental Aggregated Gradient (IAG) method proposed in [17] and analyzed in [72], where the index  $j$  of the component function updated at every iteration is chosen cyclically. Interestingly, randomizing this choice yields good practical benefits.
