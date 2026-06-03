

---

## Page 1

# SPECTRAL NORMALIZATION FOR GENERATIVE ADVERSARIAL NETWORKS

Takeru Miyato<sup>1</sup>, Toshiki Kataoka<sup>1</sup>, Masanori Koyama<sup>2</sup>, Yuichi Yoshida<sup>3</sup>

{miyato, kataoka}@preferred.jp

koyama.masanori@gmail.com

yyoshida@nii.ac.jp

<sup>1</sup>Preferred Networks, Inc. <sup>2</sup>Ritsumeikan University <sup>3</sup>National Institute of Informatics

## ABSTRACT

One of the challenges in the study of generative adversarial networks is the instability of its training. In this paper, we propose a novel weight normalization technique called spectral normalization to stabilize the training of the discriminator. Our new normalization technique is computationally light and easy to incorporate into existing implementations. We tested the efficacy of spectral normalization on CIFAR10, STL-10, and ILSVRC2012 dataset, and we experimentally confirmed that spectrally normalized GANs (SN-GANs) is capable of generating images of better or equal quality relative to the previous training stabilization techniques. The code with Chainer (Tokui et al., 2015), generated images and pretrained models are available at [https://github.com/pfnet-research/sngan\\_projection](https://github.com/pfnet-research/sngan_projection).

## 1 INTRODUCTION

*Generative adversarial networks* (GANs) (Goodfellow et al., 2014) have been enjoying considerable success as a framework of generative models in recent years, and it has been applied to numerous types of tasks and datasets (Radford et al., 2016; Salimans et al., 2016; Ho & Ermon, 2016; Li et al., 2017). In a nutshell, GANs are a framework to produce a model distribution that mimics a given target distribution, and it consists of a generator that produces the model distribution and a discriminator that distinguishes the model distribution from the target. The concept is to consecutively train the model distribution and the discriminator in turn, with the goal of reducing the difference between the model distribution and the target distribution measured by the best discriminator possible at each step of the training. GANs have been drawing attention in the machine learning community not only for its ability to learn highly structured probability distribution but also for its theoretically interesting aspects. For example, (Nowozin et al., 2016; Uehara et al., 2016; Mohamed & Lakshminarayanan, 2017) revealed that the training of the discriminator amounts to the training of a good estimator for the density ratio between the model distribution and the target. This is a perspective that opens the door to the methods of *implicit models* (Mohamed & Lakshminarayanan, 2017; Tran et al., 2017) that can be used to carry out variational optimization without the direct knowledge of the density function.

A persisting challenge in the training of GANs is the performance control of the discriminator. In high dimensional spaces, the density ratio estimation by the discriminator is often inaccurate and unstable during the training, and generator networks fail to learn the multimodal structure of the target distribution. Even worse, when the support of the model distribution and the support of the target distribution are disjoint, there exists a discriminator that can perfectly distinguish the model distribution from the target (Arjovsky & Bottou, 2017). Once such discriminator is produced in this situation, the training of the generator comes to complete stop, because the derivative of the so-produced discriminator with respect to the input turns out to be 0. This motivates us to introduce some form of restriction to the choice of the discriminator.

In this paper, we propose a novel weight normalization method called *spectral normalization* that can stabilize the training of discriminator networks. Our normalization enjoys following favorable properties.


---

## Page 2

- • Lipschitz constant is the only hyper-parameter to be tuned, and the algorithm does not require intensive tuning of the only hyper-parameter for satisfactory performance.
- • Implementation is simple and the additional computational cost is small.

In fact, our normalization method also functioned well even without tuning Lipschitz constant, which is the only hyper parameter. In this study, we provide explanations of the effectiveness of spectral normalization for GANs against other regularization techniques, such as weight normalization (Salimans & Kingma, 2016), weight clipping (Arjovsky et al., 2017), and gradient penalty (Gulrajani et al., 2017). We also show that, in the absence of complimentary regularization techniques (e.g., batch normalization, weight decay and feature matching on the discriminator), spectral normalization can improve the sheer quality of the generated images better than weight normalization and gradient penalty.

## 2 METHOD

In this section, we will lay the theoretical groundwork for our proposed method. Let us consider a simple discriminator made of a neural network of the following form, with the input  $\mathbf{x}$ :

$$f(\mathbf{x}, \theta) = W^{L+1} a_L(W^L (a_{L-1}(W^{L-1}(\dots a_1(W^1 \mathbf{x}) \dots)))), \quad (1)$$

where  $\theta := \{W^1, \dots, W^L, W^{L+1}\}$  is the learning parameters set,  $W^l \in \mathbb{R}^{d_l \times d_{l-1}}$ ,  $W^{L+1} \in \mathbb{R}^{1 \times d_L}$ , and  $a_l$  is an element-wise non-linear activation function. We omit the bias term of each layer for simplicity. The final output of the discriminator is given by

$$D(\mathbf{x}, \theta) = \mathcal{A}(f(\mathbf{x}, \theta)), \quad (2)$$

where  $\mathcal{A}$  is an activation function corresponding to the divergence of distance measure of the user's choice. The standard formulation of GANs is given by

$$\min_G \max_D V(G, D)$$

where  $\min$  and  $\max$  of  $G$  and  $D$  are taken over the set of generator and discriminator functions, respectively. The conventional form of  $V(G, D)$  (Goodfellow et al., 2014) is given by  $\mathbb{E}_{\mathbf{x} \sim q_{\text{data}}} [\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{x}' \sim p_G} [\log(1 - D(\mathbf{x}'))]$ , where  $q_{\text{data}}$  is the data distribution and  $p_G$  is the (model) generator distribution to be learned through the adversarial min-max optimization. The activation function  $\mathcal{A}$  that is used in the  $D$  of this expression is some continuous function with range  $[0, 1]$  (e.g, sigmoid function). It is known that, for a fixed generator  $G$ , the optimal discriminator for this form of  $V(G, D)$  is given by  $D_G^*(\mathbf{x}) := q_{\text{data}}(\mathbf{x}) / (q_{\text{data}}(\mathbf{x}) + p_G(\mathbf{x}))$ .

The machine learning community has been pointing out recently that the function space from which the discriminators are selected crucially affects the performance of GANs. A number of works (Uehara et al., 2016; Qi, 2017; Gulrajani et al., 2017) advocate the importance of Lipschitz continuity in assuring the boundedness of statistics. For example, the optimal discriminator of GANs on the above standard formulation takes the form

$$D_G^*(\mathbf{x}) = \frac{q_{\text{data}}(\mathbf{x})}{q_{\text{data}}(\mathbf{x}) + p_G(\mathbf{x})} = \text{sigmoid}(f^*(\mathbf{x})), \text{ where } f^*(\mathbf{x}) = \log q_{\text{data}}(\mathbf{x}) - \log p_G(\mathbf{x}), \quad (3)$$

and its derivative

$$\nabla_{\mathbf{x}} f^*(\mathbf{x}) = \frac{1}{q_{\text{data}}(\mathbf{x})} \nabla_{\mathbf{x}} q_{\text{data}}(\mathbf{x}) - \frac{1}{p_G(\mathbf{x})} \nabla_{\mathbf{x}} p_G(\mathbf{x}) \quad (4)$$

can be unbounded or even incomputable. This prompts us to introduce some regularity condition to the derivative of  $f(\mathbf{x})$ .

A particularly successful works in this array are (Qi, 2017; Arjovsky et al., 2017; Gulrajani et al., 2017), which proposed methods to control the Lipschitz constant of the discriminator by adding regularization terms defined on input examples  $\mathbf{x}$ . We would follow their footsteps and search for the discriminator  $D$  from the set of  $K$ -Lipschitz continuous functions, that is,

$$\arg \max_{\|f\|_{\text{Lip}} \leq K} V(G, D), \quad (5)$$


---

## Page 3

where we mean by  $\|f\|_{\text{Lip}}$  the smallest value  $M$  such that  $\|f(\mathbf{x}) - f(\mathbf{x}')\|/\|\mathbf{x} - \mathbf{x}'\| \leq M$  for any  $\mathbf{x}, \mathbf{x}'$ , with the norm being the  $\ell_2$  norm.

While input based regularizations allow for relatively easy formulations based on samples, they also suffer from the fact that, they cannot impose regularization on the space outside of the supports of the generator and data distributions without introducing somewhat heuristic means. A method we would introduce in this paper, called *spectral normalization*, is a method that aims to skirt this issue by *normalizing* the weight matrices using the technique devised by Yoshida & Miyato (2017).

## 2.1 SPECTRAL NORMALIZATION

Our spectral normalization controls the Lipschitz constant of the discriminator function  $f$  by literally constraining the spectral norm of each layer  $g : \mathbf{h}_{in} \mapsto \mathbf{h}_{out}$ . By definition, Lipschitz norm  $\|g\|_{\text{Lip}}$  is equal to  $\sup_{\mathbf{h}} \sigma(\nabla g(\mathbf{h}))$ , where  $\sigma(A)$  is the spectral norm of the matrix  $A$  ( $L_2$  matrix norm of  $A$ )

$$\sigma(A) := \max_{\mathbf{h}: \mathbf{h} \neq \mathbf{0}} \frac{\|A\mathbf{h}\|_2}{\|\mathbf{h}\|_2} = \max_{\|\mathbf{h}\|_2 \leq 1} \|A\mathbf{h}\|_2, \quad (6)$$

which is equivalent to the largest singular value of  $A$ . Therefore, for a linear layer  $g(\mathbf{h}) = W\mathbf{h}$ , the norm is given by  $\|g\|_{\text{Lip}} = \sup_{\mathbf{h}} \sigma(\nabla g(\mathbf{h})) = \sup_{\mathbf{h}} \sigma(W) = \sigma(W)$ . If the Lipschitz norm of the activation function  $\|a_l\|_{\text{Lip}}$  is equal to 1<sup>1</sup>, we can use the inequality  $\|g_1 \circ g_2\|_{\text{Lip}} \leq \|g_1\|_{\text{Lip}} \cdot \|g_2\|_{\text{Lip}}$  to observe the following bound on  $\|f\|_{\text{Lip}}$ :

$$\begin{aligned} \|f\|_{\text{Lip}} &\leq \|(\mathbf{h}_L \mapsto W^{L+1}\mathbf{h}_L)\|_{\text{Lip}} \cdot \|a_L\|_{\text{Lip}} \cdot \|(\mathbf{h}_{L-1} \mapsto W^L\mathbf{h}_{L-1})\|_{\text{Lip}} \\ &\cdots \|a_1\|_{\text{Lip}} \cdot \|(\mathbf{h}_0 \mapsto W^1\mathbf{h}_0)\|_{\text{Lip}} = \prod_{l=1}^{L+1} \|(\mathbf{h}_{l-1} \mapsto W^l\mathbf{h}_{l-1})\|_{\text{Lip}} = \prod_{l=1}^{L+1} \sigma(W^l). \end{aligned} \quad (7)$$

Our *spectral normalization* normalizes the spectral norm of the weight matrix  $W$  so that it satisfies the Lipschitz constraint  $\sigma(W) = 1$ :

$$\bar{W}_{\text{SN}}(W) := W/\sigma(W). \quad (8)$$

If we normalize each  $W^l$  using (8), we can appeal to the inequality (7) and the fact that  $\sigma(\bar{W}_{\text{SN}}(W)) = 1$  to see that  $\|f\|_{\text{Lip}}$  is bounded from above by 1.

Here, we would like to emphasize the difference between our spectral normalization and spectral norm "regularization" introduced by Yoshida & Miyato (2017). Unlike our method, spectral norm "regularization" penalizes the spectral norm by adding explicit regularization term to the objective function. Their method is fundamentally different from our method in that they do not make an attempt to 'set' the spectral norm to a designated value. Moreover, when we reorganize the derivative of our normalized cost function and rewrite our objective function (12), we see that our method is augmenting the cost function with a sample data *dependent* regularization function. Spectral norm regularization, on the other hand, imposes sample data *independent* regularization on the cost function, just like L2 regularization and Lasso.

## 2.2 FAST APPROXIMATION OF THE SPECTRAL NORM $\sigma(W)$

As we mentioned above, the spectral norm  $\sigma(W)$  that we use to regularize each layer of the discriminator is the largest singular value of  $W$ . If we naively apply singular value decomposition to compute the  $\sigma(W)$  at each round of the algorithm, the algorithm can become computationally heavy. Instead, we can use the power iteration method to estimate  $\sigma(W)$  (Golub & Van der Vorst, 2000; Yoshida & Miyato, 2017). With power iteration method, we can estimate the spectral norm with very small additional computational time relative to the full computational cost of the vanilla GANs. Please see Appendix A for the detail method and Algorithm 1 for the summary of the actual spectral normalization algorithm.

<sup>1</sup>For examples, ReLU (Jarrett et al., 2009; Nair & Hinton, 2010; Glorot et al., 2011) and leaky ReLU (Maas et al., 2013) satisfies the condition, and many popular activation functions satisfy  $K$ -Lipschitz constraint for some predefined  $K$  as well.


---

## Page 4

### 2.3 GRADIENT ANALYSIS OF THE SPECTRALLY NORMALIZED WEIGHTS

The gradient<sup>2</sup> of  $\bar{W}_{\text{SN}}(W)$  with respect to  $W_{ij}$  is:

$$\frac{\partial \bar{W}_{\text{SN}}(W)}{\partial W_{ij}} = \frac{1}{\sigma(W)} E_{ij} - \frac{1}{\sigma(W)^2} \frac{\partial \sigma(W)}{\partial W_{ij}} W = \frac{1}{\sigma(W)} E_{ij} - \frac{[\mathbf{u}_1 \mathbf{v}_1^T]_{ij}}{\sigma(W)^2} W \quad (9)$$

$$= \frac{1}{\sigma(W)} (E_{ij} - [\mathbf{u}_1 \mathbf{v}_1^T]_{ij} \bar{W}_{\text{SN}}), \quad (10)$$

where  $E_{ij}$  is the matrix whose  $(i, j)$ -th entry is 1 and zero everywhere else, and  $\mathbf{u}_1$  and  $\mathbf{v}_1$  are respectively the first left and right singular vectors of  $W$ . If  $\mathbf{h}$  is the hidden layer in the network to be transformed by  $\bar{W}_{\text{SN}}$ , the derivative of the  $V(G, D)$  calculated over the mini-batch with respect to  $W$  of the discriminator  $D$  is given by:

$$\frac{\partial V(G, D)}{\partial W} = \frac{1}{\sigma(W)} (\hat{\mathbf{E}} [\boldsymbol{\delta} \mathbf{h}^T] - (\hat{\mathbf{E}} [\boldsymbol{\delta}^T \bar{W}_{\text{SN}} \mathbf{h}]) \mathbf{u}_1 \mathbf{v}_1^T) \quad (11)$$

$$= \frac{1}{\sigma(W)} (\hat{\mathbf{E}} [\boldsymbol{\delta} \mathbf{h}^T] - \lambda \mathbf{u}_1 \mathbf{v}_1^T) \quad (12)$$

where  $\boldsymbol{\delta} := (\partial V(G, D) / \partial (\bar{W}_{\text{SN}} \mathbf{h}))^T$ ,  $\lambda := \hat{\mathbf{E}} [\boldsymbol{\delta}^T (\bar{W}_{\text{SN}} \mathbf{h})]$ , and  $\hat{\mathbf{E}}[\cdot]$  represents empirical expectation over the mini-batch.  $\frac{\partial V}{\partial W} = 0$  when  $\hat{\mathbf{E}}[\boldsymbol{\delta} \mathbf{h}^T] = k \mathbf{u}_1 \mathbf{v}_1^T$  for some  $k \in \mathbb{R}$ .

We would like to comment on the implication of (12). The first term  $\hat{\mathbf{E}} [\boldsymbol{\delta} \mathbf{h}^T]$  is the same as the derivative of the weights without normalization. In this light, the second term in the expression can be seen as the regularization term penalizing the first singular components with the *adaptive* regularization coefficient  $\lambda$ .  $\lambda$  is positive when  $\boldsymbol{\delta}$  and  $\bar{W}_{\text{SN}} \mathbf{h}$  are pointing in similar direction, and this prevents the column space of  $W$  from concentrating into one particular direction in the course of the training. In other words, spectral normalization prevents the transformation of each layer from becoming to sensitive in one direction. We can also use spectral normalization to devise a new parametrization for the model. Namely, we can split the layer map into two separate trainable components: spectrally normalized map and the spectral norm constant. As it turns out, this parametrization has its merit on its own and promotes the performance of GANs (See Appendix E).

### 3 SPECTRAL NORMALIZATION VS OTHER REGULARIZATION TECHNIQUES

The weight normalization introduced by Salimans & Kingma (2016) is a method that normalizes the  $\ell_2$  norm of each row vector in the weight matrix. Mathematically, this is equivalent to requiring the weight by the weight normalization  $\bar{W}_{\text{WN}}$ :

$$\sigma_1(\bar{W}_{\text{WN}})^2 + \sigma_2(\bar{W}_{\text{WN}})^2 + \dots + \sigma_T(\bar{W}_{\text{WN}})^2 = d_o, \text{ where } T = \min(d_i, d_o), \quad (13)$$

where  $\sigma_t(A)$  is a  $t$ -th singular value of matrix  $A$ . Therefore, up to a scaler, this is same as the Frobenius normalization, which requires the sum of the squared singular values to be 1. These normalizations, however, inadvertently impose much stronger constraint on the matrix than intended. If  $\bar{W}_{\text{WN}}$  is the weight normalized matrix of dimension  $d_i \times d_o$ , the norm  $\|\bar{W}_{\text{WN}} \mathbf{h}\|_2$  for a fixed unit vector  $\mathbf{h}$  is maximized at  $\|\bar{W}_{\text{WN}} \mathbf{h}\|_2 = \sqrt{d_o}$  when  $\sigma_1(\bar{W}_{\text{WN}}) = \sqrt{d_o}$  and  $\sigma_t(\bar{W}_{\text{WN}}) = 0$  for  $t = 2, \dots, T$ , which means that  $\bar{W}_{\text{WN}}$  is of rank one. Similar thing can be said to the Frobenius normalization (See the appendix for more details). Using such  $\bar{W}_{\text{WN}}$  corresponds to using only one feature to discriminate the model probability distribution from the target. In order to retain as much norm of the input as possible and hence to make the discriminator more sensitive, one would hope to make the norm of  $\bar{W}_{\text{WN}} \mathbf{h}$  large. For weight normalization, however, this comes at the cost of reducing the rank and hence the number of features to be used for the discriminator. Thus, there is a conflict of interests between weight normalization and our desire to use as many features as possible to distinguish the generator distribution from the target distribution. The former interest often reigns over the other in many cases, inadvertently diminishing the number of features to be used by the discriminators. Consequently, the algorithm would produce a rather arbitrary model distribution

<sup>2</sup>Indeed, when the spectrum has multiplicities, we would be looking at subgradients here. However, the probability of this happening is zero (almost surely), so we would continue discussions without giving considerations to such events.


---

## Page 5

that matches the target distribution only at select few features. Weight clipping (Arjovsky et al., 2017) also suffers from same pitfall.

Our spectral normalization, on the other hand, do not suffer from such a conflict in interest. Note that the Lipschitz constant of a linear operator is determined only by the maximum singular value. In other words, the spectral norm is independent of rank. Thus, unlike the weight normalization, our spectral normalization allows the parameter matrix to use as many features as possible while satisfying local 1-Lipschitz constraint. Our spectral normalization leaves more freedom in choosing the number of singular components (features) to feed to the next layer of the discriminator.

Brock et al. (2016) introduced orthonormal regularization on each weight to stabilize the training of GANs. In their work, Brock et al. (2016) augmented the adversarial objective function by adding the following term:

$$\|W^T W - I\|_F^2. \quad (14)$$

While this seems to serve the same purpose as spectral normalization, orthonormal regularization are mathematically quite different from our spectral normalization because the orthonormal regularization destroys the information about the spectrum by setting all the singular values to one. On the other hand, spectral normalization only scales the spectrum so that the its maximum will be one.

Gulrajani et al. (2017) used Gradient penalty method in combination with WGAN. In their work, they placed  $K$ -Lipschitz constant on the discriminator by augmenting the objective function with the regularizer that rewards the function for having local 1-Lipschitz constant (i.e.  $\|\nabla_{\hat{x}} f\|_2 = 1$ ) at discrete sets of points of the form  $\hat{x} := \epsilon \tilde{x} + (1 - \epsilon)x$  generated by interpolating a sample  $\tilde{x}$  from generative distribution and a sample  $x$  from the data distribution. While this rather straightforward approach does not suffer from the problems we mentioned above regarding the effective dimension of the feature space, the approach has an obvious weakness of being heavily dependent on the support of the current generative distribution. As a matter of course, the generative distribution and its support gradually changes in the course of the training, and this can destabilize the effect of such regularization. In fact, we empirically observed that a high learning rate can destabilize the performance of WGAN-GP. On the contrary, our spectral normalization regularizes the function the operator space, and the effect of the regularization is more stable with respect to the choice of the batch. Training with our spectral normalization does not easily destabilize with aggressive learning rate. Moreover, WGAN-GP requires more computational cost than our spectral normalization with single-step power iteration, because the computation of  $\|\nabla_{\hat{x}} f\|_2$  requires one whole round of forward and backward propagation. In the appendix section, we compare the computational cost of the two methods for the same number of updates.

## 4 EXPERIMENTS

In order to evaluate the efficacy of our approach and investigate the reason behind its efficacy, we conducted a set of extensive experiments of unsupervised image generation on CIFAR-10 (Torralba et al., 2008) and STL-10 (Coates et al., 2011), and compared our method against other normalization techniques. To see how our method fares against large dataset, we also applied our method on ILSVRC2012 dataset (ImageNet) (Russakovsky et al., 2015) as well. This section is structured as follows. First, we will discuss the objective functions we used to train the architecture, and then we will describe the optimization settings we used in the experiments. We will then explain two performance measures on the images to evaluate the images produced by the trained generators. Finally, we will summarize our results on CIFAR-10, STL-10, and ImageNet.

As for the architecture of the discriminator and generator, we used convolutional neural networks. Also, for the evaluation of the spectral norm for the convolutional weight  $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}} \times h \times w}$ , we treated the operator as a 2-D matrix of dimension  $d_{\text{out}} \times (d_{\text{in}}hw)^3$ . We trained the parameters of the generator with batch normalization (Ioffe & Szegedy, 2015). We refer the readers to Table 3 in the appendix section for more details of the architectures.

<sup>3</sup>Note that, since we are conducting the convolution discretely, the spectral norm will depend on the size of the stride and padding. However, the answer will only differ by some predefined  $K$ .


---

## Page 6

For all methods other than WGAN-GP, we used the following standard objective function for the adversarial loss:

$$V(G, D) := \mathbb{E}_{\mathbf{x} \sim q_{\text{data}}(\mathbf{x})} [\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})} [\log(1 - D(G(\mathbf{z})))], \quad (15)$$

where  $\mathbf{z} \in \mathbb{R}^{d_z}$  is a latent variable,  $p(\mathbf{z})$  is the standard normal distribution  $\mathcal{N}(0, I)$ , and  $G : \mathbb{R}^{d_z} \rightarrow \mathbb{R}^{d_o}$  is a deterministic generator function. We set  $d_z$  to 128 for all of our experiments. For the updates of  $G$ , we used the alternate cost proposed by Goodfellow et al. (2014)  $-\mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})} [\log(D(G(\mathbf{z})))]$  as used in Goodfellow et al. (2014) and Warde-Farley & Bengio (2017). For the updates of  $D$ , we used the original cost defined in (15). We also tested the performance of the algorithm with the so-called hinge loss, which is given by

$$V_D(\hat{G}, D) = \mathbb{E}_{\mathbf{x} \sim q_{\text{data}}(\mathbf{x})} [\min(0, -1 + D(\mathbf{x}))] + \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})} [\min(0, -1 - D(\hat{G}(\mathbf{z})))] \quad (16)$$

$$V_G(G, \hat{D}) = -\mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})} [\hat{D}(G(\mathbf{z}))], \quad (17)$$

respectively for the discriminator and the generator. Optimizing these objectives is equivalent to minimizing the so-called reverse KL divergence :  $\text{KL}[p_g || q_{\text{data}}]$ . This type of loss has been already proposed and used in Lim & Ye (2017); Tran et al. (2017). The algorithm based on the hinge loss also showed good performance when evaluated with inception score and FID. For Wasserstein GANs with gradient penalty (WGAN-GP) (Gulrajani et al., 2017), we used the following objective function:  $V(G, D) := \mathbb{E}_{\mathbf{x} \sim q_{\text{data}}} [D(\mathbf{x})] - \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})} [D(G(\mathbf{z}))] - \lambda \mathbb{E}_{\hat{\mathbf{x}} \sim p_{\hat{\mathbf{x}}}} [\|\nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}})\|_2 - 1]^2$ , where the regularization term is the one we introduced in the appendix section D.4.

For quantitative assessment of generated examples, we used *inception score* (Salimans et al., 2016) and *Fréchet inception distance* (FID) (Heusel et al., 2017). Please see Appendix B.1 for the details of each score.

#### 4.1 RESULTS ON CIFAR10 AND STL-10

In this section, we report the accuracy of the spectral normalization (we use the abbreviation: SN-GAN for the spectrally normalized GANs) during the training, and the dependence of the algorithm’s performance on the hyperparameters of the optimizer. We also compare the performance quality of the algorithm against those of other regularization/normalization techniques for the discriminator networks, including: Weight clipping (Arjovsky et al., 2017), WGAN-GP (Gulrajani et al., 2017), batch-normalization (BN) (Ioffe & Szegedy, 2015), layer normalization (LN) (Ba et al., 2016), weight normalization (WN) (Salimans & Kingma, 2016) and orthonormal regularization (*orthonormal*) (Brock et al., 2016). In order to evaluate the stand-alone efficacy of the gradient penalty, we also applied the gradient penalty term to the standard adversarial loss of GANs (15). We would refer to this method as ‘GAN-GP’. For weight clipping, we followed the original work Arjovsky et al. (2017) and set the clipping constant  $c$  at 0.01 for the convolutional weight of each layer. For gradient penalty, we set  $\lambda$  to 10, as suggested in Gulrajani et al. (2017). For *orthonormal*, we initialized the each weight of  $D$  with a randomly selected orthonormal operator and trained GANs with the objective function augmented with the regularization term used in Brock et al. (2016). For all comparative studies throughout, we excluded the multiplier parameter  $\gamma$  in the weight normalization method, as well as in batch normalization and layer normalization method. This was done in order to prevent the methods from overtly violating the Lipschitz condition. When we experimented with different multiplier parameter, we were in fact not able to achieve any improvement.

For optimization, we used the Adam optimizer Kingma & Ba (2015) in all of our experiments. We tested with 6 settings for (1)  $n_{\text{dis}}$ , the number of updates of the discriminator per one update of the generator and (2) learning rate  $\alpha$  and the first and second order momentum parameters ( $\beta_1, \beta_2$ ) of Adam. We list the details of these settings in Table 1 in the appendix section. Out of these 6 settings, A, B, and C are the settings used in previous representative works. The purpose of the settings D, E, and F is to evaluate the performance of the algorithms implemented with more aggressive learning rates. For the details of the architectures of convolutional networks deployed in the generator and the discriminator, we refer the readers to Table 3 in the appendix section. The number of updates for GAN generator were 100K for all experiments, unless otherwise noted.

Firstly, we inspected the spectral norm of each layer during the training to make sure that our spectral normalization procedure is indeed serving its purpose. As we can see in the Figure 9 in the C.1,


---

## Page 7

Table 1: Hyper-parameter settings we tested in our experiments.  $\dagger$ ,  $\ddagger$  and  $\star$  are the hyperparameter settings following Gulrajani et al. (2017), Warde-Farley & Bengio (2017) and Radford et al. (2016), respectively.

<table border="1">
<thead>
<tr>
<th>Setting</th>
<th><math>\alpha</math></th>
<th><math>\beta_1</math></th>
<th><math>\beta_2</math></th>
<th><math>n_{dis}</math></th>
</tr>
</thead>
<tbody>
<tr>
<td>A<math>^\dagger</math></td>
<td>0.0001</td>
<td>0.5</td>
<td>0.9</td>
<td>5</td>
</tr>
<tr>
<td>B<math>^{\ddagger}</math></td>
<td>0.0001</td>
<td>0.5</td>
<td>0.999</td>
<td>1</td>
</tr>
<tr>
<td>C<math>^\star</math></td>
<td>0.0002</td>
<td>0.5</td>
<td>0.999</td>
<td>1</td>
</tr>
<tr>
<td>D</td>
<td>0.001</td>
<td>0.5</td>
<td>0.9</td>
<td>5</td>
</tr>
<tr>
<td>E</td>
<td>0.001</td>
<td>0.5</td>
<td>0.999</td>
<td>5</td>
</tr>
<tr>
<td>F</td>
<td>0.001</td>
<td>0.9</td>
<td>0.999</td>
<td>5</td>
</tr>
</tbody>
</table>

![Figure 1: Inception scores on CIFAR-10 and STL-10 with different methods and hyperparameters. (a) CIFAR-10: Methods compared are Weight clip., GAN-GP, WGAN-GP, BN, LN, WN, Orthonormal, and SN. (b) STL-10: Methods compared are Weight clip., WGAN-GP, LN, WN, Orthonormal, and SN. For each method, six bars represent hyperparameter settings A (dark blue), B (blue), C (light blue), D (orange), E (red), and F (dark red). In both datasets, SN generally achieves the highest scores, followed by Orthonormal and LN. WGAN-GP and GAN-GP show lower scores, while Weight clip. and BN perform poorly. Setting A consistently yields the highest scores across all methods and datasets.](01f240f574d2167dfdb5ddf40a5607a8_4_img.webp)

Figure 1: Inception scores on CIFAR-10 and STL-10 with different methods and hyperparameters (higher is better).

the spectral norms of these layers floats around 1–1.05 region throughout the training. Please see Appendix C.1 for more details.

In Figures 1 and 2 we show the inception scores of each method with the settings A–F. We can see that spectral normalization is relatively robust with aggressive learning rates and momentum parameters. WGAN-GP fails to train good GANs at high learning rates and high momentum parameters on both CIFAR-10 and STL-10. Orthonormal regularization performed poorly for the setting E on the STL-10, but performed slightly better than our method with the optimal setting. These results suggest that our method is more robust than other methods with respect to the change in the setting of the training. Also, the optimal performance of weight normalization was inferior to both WGAN-GP and spectral normalization on STL-10, which consists of more diverse examples than CIFAR-10. Best scores of spectral normalization are better than almost all other methods on both CIFAR-10 and STL-10.

In Tables 2, we show the inception scores of the different methods with optimal settings on CIFAR-10 and STL-10 dataset. We see that SN-GANs performed better than almost all contemporaries on the optimal settings. SN-GANs performed even better with hinge loss (17).<sup>4</sup> For the training with same number of iterations, SN-GANs fell behind orthonormal regularization for STL-10. For more detailed comparison between orthonormal regularization and spectral normalization, please see section 4.1.2.

In Figure 6 we show the images produced by the generators trained with WGAN-GP, weight normalization, and spectral normalization. SN-GANs were consistently better than GANs with weight normalization in terms of the quality of generated images. To be more precise, as we mentioned in Section 3, the set of images generated by spectral normalization was clearer and more diverse than the images produced by the weight normalization. We can also see that WGAN-GP failed to train good GANs with high learning rates and high momentums (D, E and F). The generated images

<sup>4</sup>As for STL-10, we also ran SN-GANs over twice time longer iterations because it did not seem to converge. Yet still, this elongated training sequence still completes before WGAN-GP with original iteration size because the optimal setting of SN-GANs (setting B,  $n_{dis} = 1$ ) is computationally light.


---

## Page 8

![Figure 2: FIDs on CIFAR-10 and STL-10 with different methods and hyperparameters. (a) CIFAR-10: Bar chart showing FID for methods Weight clip, GAN-GP, WGAN-GP, BN, LN, WN, Orthonormal, and SN across hyperparameters A, B, C, D, E, F. (b) STL-10: Bar chart showing FID for methods Weight clip, WGAN-GP, LN, WN, Orthonormal, and SN across hyperparameters A, B, C, D, E, F. In both charts, FID is on a log scale from 10^1 to 10^2. Lower FID is better. The legend indicates hyperparameters A (dark blue), B (blue), C (light blue), D (yellow), E (red), and F (dark red).](eeb68878b97cfcc7bd4751dbd1a0b7cf_2_img.webp)

Figure 2: FIDs on CIFAR-10 and STL-10 with different methods and hyperparameters (lower is better).

Table 2: Inception scores and FIDs with unsupervised image generation on CIFAR-10. † (Radford et al., 2016) (experimented by Yang et al. (2017)), ‡ (Yang et al., 2017), \* (Warde-Farley & Bengio, 2017), †† (Gulrajani et al., 2017)

<table border="1">
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="2">Inception score</th>
<th colspan="2">FID</th>
</tr>
<tr>
<th>CIFAR-10</th>
<th>STL-10</th>
<th>CIFAR-10</th>
<th>STL-10</th>
</tr>
</thead>
<tbody>
<tr>
<td>Real data</td>
<td>11.24±.12</td>
<td>26.08±.26</td>
<td>7.8</td>
<td>7.9</td>
</tr>
<tr>
<td colspan="5"><b>-Standard CNN-</b></td>
</tr>
<tr>
<td>Weight clipping</td>
<td>6.41±.11</td>
<td>7.57±.10</td>
<td>42.6</td>
<td>64.2</td>
</tr>
<tr>
<td>GAN-GP</td>
<td>6.93±.08</td>
<td></td>
<td>37.7</td>
<td></td>
</tr>
<tr>
<td>WGAN-GP</td>
<td>6.68±.06</td>
<td>8.42±.13</td>
<td>40.2</td>
<td>55.1</td>
</tr>
<tr>
<td>Batch Norm.</td>
<td>6.27±.10</td>
<td></td>
<td>56.3</td>
<td></td>
</tr>
<tr>
<td>Layer Norm.</td>
<td>7.19±.12</td>
<td>7.61±.12</td>
<td>33.9</td>
<td>75.6</td>
</tr>
<tr>
<td>Weight Norm.</td>
<td>6.84±.07</td>
<td>7.16±.10</td>
<td>34.7</td>
<td>73.4</td>
</tr>
<tr>
<td>Orthonormal</td>
<td>7.40±.12</td>
<td>8.56±.07</td>
<td>29.0</td>
<td>46.7</td>
</tr>
<tr>
<td>(ours) SN-GANs</td>
<td>7.42±.08</td>
<td>8.28±.09</td>
<td>29.3</td>
<td>53.1</td>
</tr>
<tr>
<td>Orthonormal (2x updates)</td>
<td></td>
<td>8.67±.08</td>
<td></td>
<td>44.2</td>
</tr>
<tr>
<td>(ours) SN-GANs (2x updates)</td>
<td></td>
<td>8.69±.09</td>
<td></td>
<td>47.5</td>
</tr>
<tr>
<td>(ours) SN-GANs, Eq.(17)</td>
<td>7.58±.12</td>
<td></td>
<td>25.5</td>
<td></td>
</tr>
<tr>
<td>(ours) SN-GANs, Eq.(17) (2x updates)</td>
<td></td>
<td>8.79±.14</td>
<td></td>
<td>43.2</td>
</tr>
<tr>
<td colspan="5"><b>-ResNet<sup>-5</sup></b></td>
</tr>
<tr>
<td>Orthonormal, Eq.(17)</td>
<td>7.92±.04</td>
<td>8.72±.06</td>
<td>23.8±.58</td>
<td>42.4±.99</td>
</tr>
<tr>
<td>(ours) SN-GANs, Eq.(17)</td>
<td><b>8.22±.05</b></td>
<td><b>9.10±.04</b></td>
<td><b>21.7±.21</b></td>
<td><b>40.1±.50</b></td>
</tr>
<tr>
<td>DCGAN<sup>†</sup></td>
<td>6.64±.14</td>
<td>7.84±.07</td>
<td></td>
<td></td>
</tr>
<tr>
<td>LR-GANs<sup>‡</sup></td>
<td>7.17±.07</td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>Warde-Farley et al.<sup>*</sup></td>
<td>7.72±.13</td>
<td>8.51±.13</td>
<td></td>
<td></td>
</tr>
<tr>
<td>WGAN-GP (ResNet)<sup>††</sup></td>
<td>7.86±.08</td>
<td></td>
<td></td>
<td></td>
</tr>
</tbody>
</table>

with GAN-GP, batch normalization, and layer normalization is shown in Figure 12 in the appendix section.

We also compared our algorithm against multiple benchmark methods and summarized the results on the bottom half of the Table 2. We also tested the performance of our method on ResNet based GANs used in Gulrajani et al. (2017). Please note that all methods listed thereof are all different in both optimization methods and the architecture of the model. Please see Table 4 and 5 in the appendix section for the detail network architectures. Our implementation of our algorithm was able to perform better than almost all the predecessors in the performance.

<sup>5</sup>For our ResNet experiments, we trained the same architecture with multiple random seeds for weight initialization and produced models with different parameters. We then generated 5000 images 10 times and computed the average inception score for each model. The values for ResNet on the table are the mean and standard deviation of the score computed over the set of models trained with different seeds.


---

## Page 9

![Figure 3: Squared singular values of weight matrices for CIFAR-10 and STL-10 datasets. The figure consists of two rows of plots, (a) for CIFAR-10 and (b) for STL-10. Each row contains seven subplots for layers 1 through 7. Each subplot shows the squared singular values (sigma^2) on the y-axis (0.0 to 1.0) against the index of sigma on the x-axis. Three methods are compared: Weight Clipping (WC, green line), Weight Normalization (WN, blue line), and Spectral Normalization (SN, red line). In both datasets, SN shows a much broader distribution of singular values across all layers compared to WC and WN, which are highly concentrated on a few components.](1d8a3dac9a73df603585d103c1fa4a41_2_img.webp)

Figure 3: Squared singular values of weight matrices trained with different methods: Weight clipping (WC), Weight Normalization (WN) and Spectral Normalization (SN). We scaled the singular values so that the largest singular values is equal to 1. For WN and SN, we calculated singular values of the normalized weight matrices.

#### 4.1.1 ANALYSIS OF SN-GANS

**Singular values analysis on the weights of the discriminator  $D$**  In Figure 3, we show the squared singular values of the weight matrices in the final discriminator  $D$  produced by each method using the parameter that yielded the best inception score. As we predicted in Section 3, the singular values of the first to fifth layers trained with weight clipping and weight normalization concentrate on a few components. That is, the weight matrices of these layers tend to be rank deficit. On the other hand, the singular values of the weight matrices in those layers trained with spectral normalization is more broadly distributed. When the goal is to distinguish a pair of probability distributions on the low-dimensional nonlinear data manifold embedded in a high dimensional space, rank deficiencies in lower layers can be especially fatal. Outputs of lower layers have gone through only a few sets of rectified linear transformations, which means that they tend to lie on the space that is linear in most parts. Marginalizing out many features of the input distribution in such space can result in oversimplified discriminator. We can actually confirm the effect of this phenomenon on the generated images especially in Figure 6b. The images generated with spectral normalization is more diverse and complex than those generated with weight normalization.

**Training time** On CIFAR-10, SN-GANs is slightly slower than weight normalization (about 110 ~ 120% computational time), but significantly faster than WGAN-GP. As we mentioned in Section 3, WGAN-GP is slower than other methods because WGAN-GP needs to calculate the gradient of gradient norm  $\|\nabla_x D\|_2$ . For STL-10, the computational time of SN-GANs is almost the same as vanilla GANs, because the relative computational cost of the power iteration (18) is negligible when compared to the cost of forward and backward propagation on CIFAR-10 (images size of STL-10 is larger ( $48 \times 48$ )). Please see Figure 10 in the appendix section for the actual computational time.

#### 4.1.2 COMPARISON BETWEEN SN-GANS AND ORTHONORMAL REGULARIZATION

In order to highlight the difference between our spectral normalization and orthonormal regularization, we conducted an additional set of experiments. As we explained in Section 3, orthonormal regularization is different from our method in that it destroys the spectral information and puts equal emphasis on all feature dimensions, including the ones that ‘shall’ be weeded out in the training process. To see the extent of its possibly detrimental effect, we experimented by increasing the di-


---

## Page 10

![Figure 4: A line graph showing the Inception score on the STL-10 dataset as a function of the relative size of the feature map dimension. The x-axis represents the relative size of the feature map dimension, ranging from 0.5 to 8.0, with a note that the original value is 1.0. The y-axis represents the Inception score, ranging from 7.9 to 8.6. Two data series are plotted: SN-GANs (red solid line with a light red shaded region representing standard deviation) and Orthonormal (blue dashed line with a light blue shaded region). SN-GANs start at a score of approximately 8.45 at a relative size of 0.5, drop to a minimum of about 8.15 at 1.0, and then gradually increase to about 8.35 at 8.0. Orthonormal starts at a higher score of about 8.55 at 0.5, drops to a minimum of about 8.15 at 4.0, and then continues to decrease to about 7.95 at 8.0.](1100f193597aa65ea71f250daf719cc1_2_img.webp)

Figure 4: The effect on the performance on STL-10 induced by the change of the feature map dimension of the final layer. The width of the highlighted region represents standard deviation of the results over multiple seeds of weight initialization. The *orthonormal regularization* does not perform well with large feature map dimension, possibly because of its design that forces the discriminator to use all dimensions including the ones that are unnecessary. For the setting of the optimizers’ hyper-parameters, We used the setting C, which was optimal for “*orthonormal regularization*”

![Figure 5: A line graph showing the learning curves for conditional image generation on ImageNet in terms of Inception score. The x-axis represents the number of iterations in units of 10^5, ranging from 0.0 to 4.5. The y-axis represents the Inception score, ranging from 10 to 22. Two data series are plotted: SN-GANs (red solid line with square markers) and Orthonormal (blue dashed line with circle markers). Both series start at an Inception score of approximately 12.5 at 0 iterations. SN-GANs show a steady increase, reaching a peak score of about 21.5 at 4.5 iterations. Orthonormal shows a similar initial increase but plateaus around 18.5 after 2.5 iterations, with a slight dip and then a recovery to about 17.5 at 4.5 iterations.](1100f193597aa65ea71f250daf719cc1_4_img.webp)

Figure 5: Learning curves for conditional image generation in terms of Inception score for SN-GANs and GANs with *orthonormal regularization* on ImageNet.

mension of the feature space <sup>6</sup>, especially at the final layer (7th conv) for which the training with our spectral normalization prefers relatively small feature space (dimension  $< 100$ ; see Figure 3b). As for the setting of the training, we selected the parameters for which the orthonormal regularization performed optimally. The figure 4 shows the result of our experiments. As we predicted, the performance of the orthonormal regularization deteriorates as we increase the dimension of the feature maps at the final layer. Our SN-GANs, on the other hand, does not falter with this modification of the architecture. Thus, at least in this perspective, we may such that our method is more robust with respect to the change of the network architecture.

#### 4.2 IMAGE GENERATION ON IMAGENET

To show that our method remains effective on a large high dimensional dataset, we also applied our method to the training of conditional GANs on ILRSVRC2012 dataset with 1000 classes, each consisting of approximately 1300 images, which we compressed to  $128 \times 128$  pixels. Regarding the adversarial loss for conditional GANs, we used practically the same formulation used in Mirza & Osindero (2014), except that we replaced the standard GANs loss with hinge loss (17). Please see Appendix B.3 for the details of experimental settings.

<sup>6</sup>More precisely, we simply increased the input dimension and the output dimension by the same factor. In Figure 4, ‘relative size’ = 1.0 implies that the layer structure is the same as the original.


---

## Page 11

GANs without normalization and GANs with layer normalization collapsed in the beginning of training and failed to produce any meaningful images. GANs with orthonormal normalization Brock et al. (2016) and our spectral normalization, on the other hand, was able to produce images. The inception score of the orthonormal normalization however plateaued around 20Kth iterations, while SN kept improving even afterward (Figure 5.) To our knowledge, our research is the first of its kind in succeeding to produce decent images from ImageNet dataset with a *single* pair of a discriminator and a generator (Figure 7). To measure the degree of mode-collapse, we followed the footsteps of Odena et al. (2017) and computed the intra MS-SSIM Odena et al. (2017) for pairs of independently generated GANs images of each class. We see that our SN-GANs ((intra MS-SSIM)=0.101) is suffering less from the mode-collapse than AC-GANs ((intra MS-SSIM)~0.25).

To ensure that the superiority of our method is not limited within our specific setting, we also compared the performance of SN-GANs against *orthonormal regularization* on conditional GANs with *projection discriminator* (Miyato & Koyama, 2018) as well as the standard (unconditional) GANs. In our experiments, SN-GANs achieved better performance than *orthonormal regularization* for the both settings (See Figure 13 in the appendix section).

## 5 CONCLUSION

This paper proposes spectral normalization as a stabilizer of training of GANs. When we apply spectral normalization to the GANs on image generation tasks, the generated examples are more diverse than the conventional weight normalization and achieve better or comparative inception scores relative to previous studies. The method imposes global regularization on the discriminator as opposed to local regularization introduced by WGAN-GP, and can possibly used in combinations. In the future work, we would like to further investigate where our methods stand amongst other methods on more theoretical basis, and experiment our algorithm on larger and more complex datasets.

## ACKNOWLEDGMENTS

We would like to thank the members of Preferred Networks, Inc., particularly Shin-ichi Maeda, Eiichi Matsumoto, Masaki Watanabe and Keisuke Yahata for insightful comments and discussions. We also would like to thank anonymous reviewers and commenters on the OpenReview forum for insightful discussions.

## REFERENCES

Martin Arjovsky and Léon Bottou. Towards principled methods for training generative adversarial networks. In *ICLR*, 2017.

Martin Arjovsky, Soumith Chintala, and Léon Bottou. Wasserstein generative adversarial networks. In *ICML*, pp. 214–223, 2017.

Devansh Arpit, Yingbo Zhou, Bhargava U Kota, and Venu Govindaraju. Normalization propagation: A parametric technique for removing internal covariate shift in deep networks. In *ICML*, pp. 1168–1176, 2016.

Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. *arXiv preprint arXiv:1607.06450*, 2016.

Andrew Brock, Theodore Lim, James M Ritchie, and Nick Weston. Neural photo editing with introspective adversarial networks. *arXiv preprint arXiv:1609.07093*, 2016.

Adam Coates, Andrew Ng, and Honglak Lee. An analysis of single-layer networks in unsupervised feature learning. In *AISTATS*, pp. 215–223, 2011.

Harm de Vries, Florian Strub, Jérémie Mary, Hugo Larochelle, Olivier Pietquin, and Aaron C Courville. Modulating early visual processing by language. In *NIPS*, pp. 6576–6586, 2017.

DC Dowson and BV Landau. The fréchet distance between multivariate normal distributions. *Journal of Multivariate Analysis*, 12(3):450–455, 1982.

Vincent Dumoulin, Jonathon Shlens, and Manjunath Kudlur. A learned representation for artistic style. In *ICLR*, 2017.

Xavier Glorot, Antoine Bordes, and Yoshua Bengio. Deep sparse rectifier neural networks. In *AISTATS*, pp. 315–323, 2011.

Gene H Golub and Henk A Van der Vorst. Eigenvalue computation in the 20th century. *Journal of Computational and Applied Mathematics*, 123(1):35–65, 2000.

Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In *NIPS*, pp. 2672–2680, 2014.


---

## Page 12

Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron Courville. Improved training of wasserstein GANs. *arXiv preprint arXiv:1704.00028*, 2017.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *CVPR*, pp. 770–778, 2016.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Günter Klambauer, and Sepp Hochreiter. GANs trained by a two time-scale update rule converge to a nash equilibrium. *arXiv preprint arXiv:1706.08500*, 2017.

Jonathan Ho and Stefano Ermon. Generative adversarial imitation learning. In *NIPS*, pp. 4565–4573, 2016.

Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In *ICML*, pp. 448–456, 2015.

Kevin Jarrett, Koray Kavukcuoglu, Marc’Aurelio Ranzato, and Yann LeCun. What is the best multi-stage architecture for object recognition? In *ICCV*, pp. 2146–2153, 2009.

Kui Jia, Dacheng Tao, Shenghua Gao, and Xiangmin Xu. Improving training of deep neural networks via singular value bounding. In *CVPR*, 2017.

Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In *ICLR*, 2015.

Jiwei Li, Will Monroe, Tianlin Shi, Alan Ritter, and Dan Jurafsky. Adversarial learning for neural dialogue generation. In *EMNLP*, pp. 2147–2159, 2017.

Jae Hyun Lim and Jong Chul Ye. Geometric GAN. *arXiv preprint arXiv:1705.02894*, 2017.

Andrew L Maas, Awni Y Hannun, and Andrew Y Ng. Rectifier nonlinearities improve neural network acoustic models. In *ICML Workshop on Deep Learning for Audio, Speech and Language Processing*, 2013.

Mehdi Mirza and Simon Osindero. Conditional generative adversarial nets. *arXiv preprint arXiv:1411.1784*, 2014.

Takeru Miyato and Masanori Koyama. cGANs with projection discriminator. In *ICLR*, 2018.

Shakir Mohamed and Balaji Lakshminarayanan. Learning in implicit generative models. *NIPS Workshop on Adversarial Training*, 2017.

Vinod Nair and Geoffrey E Hinton. Rectified linear units improve restricted boltzmann machines. In *ICML*, pp. 807–814, 2010.

Sebastian Nowozin, Botond Cseke, and Ryota Tomioka. f-GAN: Training generative neural samplers using variational divergence minimization. In *NIPS*, pp. 271–279, 2016.

Augustus Odena, Christopher Olah, and Jonathon Shlens. Conditional image synthesis with auxiliary classifier GANs. In *ICML*, pp. 2642–2651, 2017.

Guo-Jun Qi. Loss-sensitive generative adversarial networks on lipschitz densities. *arXiv preprint arXiv:1701.06264*, 2017.

Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. In *ICLR*, 2016.

Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. Imagenet large scale visual recognition challenge. *International Journal of Computer Vision*, 115(3):211–252, 2015.

Masaki Saito, Eiichi Matsumoto, and Shunta Saito. Temporal generative adversarial nets with singular value clipping. In *ICCV*, 2017.

Tim Salimans and Diederik P Kingma. Weight normalization: A simple reparameterization to accelerate training of deep neural networks. In *NIPS*, pp. 901–909, 2016.

Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training GANs. In *NIPS*, pp. 2226–2234, 2016.

Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. In *CVPR*, pp. 1–9, 2015.

Seiya Tokui, Kenta Oono, Shohei Hido, and Justin Clayton. Chainer: a next-generation open source framework for deep learning. In *Proceedings of workshop on machine learning systems (LearningSys) in the twenty-ninth annual conference on neural information processing systems (NIPS)*, 2015.

Antonio Torralba, Rob Fergus, and William T Freeman. 80 million tiny images: A large data set for nonparametric object and scene recognition. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 30 (11):1958–1970, 2008.

Dustin Tran, Rajesh Ranganath, and David M Blei. Deep and hierarchical implicit models. *arXiv preprint arXiv:1702.08896*, 2017.

Masatoshi Uehara, Issei Sato, Masahiro Suzuki, Kotaro Nakayama, and Yutaka Matsuo. Generative adversarial nets from a density ratio estimation perspective. *NIPS Workshop on Adversarial Training*, 2016.

David Warde-Farley and Yoshua Bengio. Improving generative adversarial networks with denoising feature matching. In *ICLR*, 2017.

Sitao Xiang and Hao Li. On the effect of batch normalization and weight normalization in generative adversarial networks. *arXiv preprint arXiv:1704.03971*, 2017.

Jianwei Yang, Anitha Kannan, Dhruv Batra, and Devi Parikh. LR-GAN: Layered recursive generative adversarial networks for image generation. *ICLR*, 2017.

Yuichi Yoshida and Takeru Miyato. Spectral norm regularization for improving the generalizability of deep learning. *arXiv preprint arXiv:1705.10941*, 2017.


---

## Page 13

![Figure 6(a): CIFAR-10 generated images. A grid of 6 columns (A-F) and 3 rows (WGAN-GP, Weight Norm., Spectral Norm.). The first column shows the original dataset. Columns A-D show realistic generated images of various objects. Column E shows a noisy, high-contrast image. Column F shows a noisy, high-contrast image.](5894dba195eb354b04c7f04b20e8b99f_2_img.webp)

Figure 6(a) displays generated images for the CIFAR-10 dataset. The grid is organized with three rows representing different methods: WGAN-GP, Weight Norm., and Spectral Norm. The first column is labeled 'Dataset' and shows the original CIFAR-10 images. Columns A, B, C, and D show realistic generated images of various objects. Column E shows a noisy, high-contrast image. Column F shows a noisy, high-contrast image.

(a) CIFAR-10

![Figure 6(b): STL-10 generated images. A grid of 6 columns (A-F) and 3 rows (WGAN-GP, Weight Norm., Spectral Norm.). The first column shows the original dataset. Columns A-D show realistic generated images of various objects. Column E shows a noisy, high-contrast image. Column F shows a noisy, high-contrast image.](5894dba195eb354b04c7f04b20e8b99f_4_img.webp)

Figure 6(b) displays generated images for the STL-10 dataset. The grid is organized with three rows representing different methods: WGAN-GP, Weight Norm., and Spectral Norm. The first column is labeled 'Dataset' and shows the original STL-10 images. Columns A, B, C, and D show realistic generated images of various objects. Column E shows a noisy, high-contrast image. Column F shows a noisy, high-contrast image.

(b) STL-10

Figure 6: Generated images on different methods: WGAN-GP, weight normalization, and spectral normalization on CIFAR-10 and STL-10.


---

## Page 14

![A 5x3 grid of 128x128 pixel images generated by SN-GANs, showing various categories: Gray whale, Welsh springer spaniel, Persian cat, Tiger, Chiffonier, Fire truck, Mosque, Palace, Schooner, Daisy, Sandbar, and Pizza.](1973c0ca41077dbfe116e8c8c4cccbe8_2_img.webp)

The figure displays a 5x3 grid of 128x128 pixel images generated by SN-GANs, trained on the ILSVRC2012 dataset. The images are organized into 12 categories, each represented by a 4x4 grid of generated samples. The categories are: Gray whale, Welsh springer spaniel, Persian cat, Tiger, Chiffonier, Fire truck, Mosque, Palace, Schooner, Daisy, Sandbar, and Pizza. Each category shows a diverse set of generated images that closely resemble the corresponding real-world objects.

Figure 7: 128x128 pixel images generated by SN-GANs trained on ILSVRC2012 dataset. The inception score is  $21.1 \pm .35$ .


---

## Page 15

## A THE ALGORITHM OF SPECTRAL NORMALIZATION

Let us describe the shortcut in Section 2.1 in more detail. We begin with vectors  $\tilde{\mathbf{u}}$  that is randomly initialized for each weight. If there is no multiplicity in the dominant singular values and if  $\tilde{\mathbf{u}}$  is not orthogonal to the first left singular vectors<sup>7</sup>, we can appeal to the principle of the power method and produce the first left and right singular vectors through the following update rule:

$$\tilde{\mathbf{v}} \leftarrow W^T \tilde{\mathbf{u}} / \|W^T \tilde{\mathbf{u}}\|_2, \quad \tilde{\mathbf{u}} \leftarrow W \tilde{\mathbf{v}} / \|W \tilde{\mathbf{v}}\|_2. \quad (18)$$

We can then approximate the spectral norm of  $W$  with the pair of so-approximated singular vectors:

$$\sigma(W) \approx \tilde{\mathbf{u}}^T W \tilde{\mathbf{v}}. \quad (19)$$

If we use SGD for updating  $W$ , the change in  $W$  at each update would be small, and hence the change in its largest singular value. In our implementation, we took advantage of this fact and reused the  $\tilde{\mathbf{u}}$  computed at each step of the algorithm as the initial vector in the subsequent step. In fact, with this ‘recycle’ procedure, one round of power iteration was sufficient in the actual experiment to achieve satisfactory performance. Algorithm 1 in the appendix summarizes the computation of the spectrally normalized weight matrix  $\bar{W}$  with this approximation. Note that this procedure is very computationally cheap even in comparison to the calculation of the forward and backward propagations on neural networks. Please see Figure 10 for actual computational time with and without spectral normalization.

---

### Algorithm 1 SGD with spectral normalization

---

- • Initialize  $\tilde{\mathbf{u}}_l \in \mathcal{R}^{d_l}$  for  $l = 1, \dots, L$  with a random vector (sampled from isotropic distribution).
- • For each update and each layer  $l$ :

1. 1. Apply power iteration method to a unnormalized weight  $W^l$ :

$$\tilde{\mathbf{v}}_l \leftarrow (W^l)^T \tilde{\mathbf{u}}_l / \|(W^l)^T \tilde{\mathbf{u}}_l\|_2 \quad (20)$$

$$\tilde{\mathbf{u}}_l \leftarrow W^l \tilde{\mathbf{v}}_l / \|W^l \tilde{\mathbf{v}}_l\|_2 \quad (21)$$

1. 2. Calculate  $\bar{W}_{\text{SN}}$  with the spectral norm:

$$\bar{W}_{\text{SN}}^l(W^l) = W^l / \sigma(W^l), \quad \text{where } \sigma(W^l) = \tilde{\mathbf{u}}_l^T W^l \tilde{\mathbf{v}}_l \quad (22)$$

1. 3. Update  $W^l$  with SGD on mini-batch dataset  $\mathcal{D}_M$  with a learning rate  $\alpha$ :

$$W^l \leftarrow W^l - \alpha \nabla_{W^l} \ell(\bar{W}_{\text{SN}}^l(W^l), \mathcal{D}_M) \quad (23)$$


---

## B EXPERIMENTAL SETTINGS

### B.1 PERFORMANCE MEASURES

Inception score is introduced originally by Salimans et al. (2016):  $I(\{x_n\}_{n=1}^N) := \exp(E[D_{\text{KL}}[p(y|x)||p(y)]])$ , where  $p(y)$  is approximated by  $\frac{1}{N} \sum_{n=1}^N p(y|x_n)$  and  $p(y|x)$  is the trained Inception convolutional neural network (Szegedy et al., 2015), which we would refer to Inception model for short. In their work, Salimans et al. (2016) reported that this score is strongly correlated with subjective human judgment of image quality. Following the procedure in Salimans et al. (2016); Warde-Farley & Bengio (2017), we calculated the score for randomly generated 5000 examples from each trained generator to evaluate its ability to generate natural images. We repeated each experiment 10 times and reported the average and the standard deviation of the inception scores.

Fréchet inception distance (Heusel et al., 2017) is another measure for the quality of the generated examples that uses 2nd order information of the final layer of the inception model applied to the

<sup>7</sup>In practice, we are safe to assume that  $\tilde{\mathbf{u}}$  generated from uniform distribution on the sphere is not orthogonal to the first singular vectors, because this can happen with probability 0.


---

## Page 16

examples. On its own, the *Fréchet distance* Dowson & Landau (1982) is 2-Wasserstein distance between two distribution  $p_1$  and  $p_2$  assuming they are both multivariate Gaussian distributions:

$$F(p_1, p_2) = \|\mu_{p_1} - \mu_{p_2}\|_2^2 + \text{trace} \left( C_{p_1} + C_{p_2} - 2(C_{p_1} C_{p_2})^{1/2} \right), \quad (24)$$

where  $\{\mu_{p_1}, C_{p_1}\}, \{\mu_{p_2}, C_{p_2}\}$  are the mean and covariance of samples from  $q$  and  $p$ , respectively. If  $f_\Theta$  is the output of the final layer of the inception model before the softmax, the Fréchet inception distance (FID) between two distributions  $p_1$  and  $p_2$  on the images is the distance between  $f_\Theta \circ p_1$  and  $f_\Theta \circ p_2$ . We computed the Fréchet inception distance between the true distribution and the generated distribution empirically over 10000 and 5000 samples. Multiple repetition of the experiments did not exhibit any notable variations on this score.

## B.2 IMAGE GENERATION ON CIFAR-10 AND STL-10

For the comparative study, we experimented with the recent ResNet architecture of Gulrajani et al. (2017) as well as the standard CNN. For this additional set of experiments, we used Adam again for the optimization and used the very hyper parameter used in Gulrajani et al. (2017) ( $\alpha = 0.0002, \beta_1 = 0, \beta_2 = 0.9, n_{dis} = 5$ ). For our SN-GANs, we doubled the feature map in the generator from the original, because this modification achieved better results. Note that when we doubled the dimension of the feature map for the WGAN-GP experiment, however, the performance deteriorated.

## B.3 IMAGE GENERATION ON IMAGENET

The images used in this set of experiments were resized to  $128 \times 128$  pixels. The details of the architecture are given in Table 6. For the generator network of conditional GANs, we used conditional batch normalization (CBN) (Dumoulin et al., 2017; de Vries et al., 2017). Namely we replaced the standard batch normalization layer with the CBN conditional to the label information  $y \in \{1, \dots, 1000\}$ . For the optimization, we used Adam with the same hyperparameters we used for ResNet on CIFAR-10 and STL-10 dataset. We trained the networks with 450K generator updates, and applied linear decay for the learning rate after 400K iterations so that the rate would be 0 at the end.

## B.4 NETWORK ARCHITECTURES

Table 3: Standard CNN models for CIFAR-10 and STL-10 used in our experiments on image Generation. The slopes of all lReLU functions in the networks are set to 0.1.

<table border="1">
<tbody>
<tr>
<td><math>z \in \mathbb{R}^{128} \sim \mathcal{N}(0, I)</math></td>
<td>RGB image <math>x \in \mathbb{R}^{M \times M \times 3}</math></td>
</tr>
<tr>
<td>dense <math>\rightarrow M_g \times M_g \times 512</math></td>
<td><math>3 \times 3</math>, stride=1 conv 64 lReLU<br/><math>4 \times 4</math>, stride=2 conv 64 lReLU</td>
</tr>
<tr>
<td><math>4 \times 4</math>, stride=2 deconv. BN 256 ReLU</td>
<td><math>3 \times 3</math>, stride=1 conv 128 lReLU<br/><math>4 \times 4</math>, stride=2 conv 128 lReLU</td>
</tr>
<tr>
<td><math>4 \times 4</math>, stride=2 deconv. BN 128 ReLU</td>
<td><math>3 \times 3</math>, stride=1 conv 256 lReLU<br/><math>4 \times 4</math>, stride=2 conv 256 lReLU</td>
</tr>
<tr>
<td><math>4 \times 4</math>, stride=2 deconv. BN 64 ReLU</td>
<td><math>3 \times 3</math>, stride=1 conv. 512 lReLU</td>
</tr>
<tr>
<td><math>3 \times 3</math>, stride=1 conv. 3 Tanh</td>
<td>dense <math>\rightarrow 1</math></td>
</tr>
</tbody>
</table>

(a) Generator,  $M_g = 4$  for SVHN and CIFAR10, and  $M_g = 6$  for STL-10

(b) Discriminator,  $M = 32$  for SVHN and CIFAR10, and  $M = 48$  for STL-10


---

## Page 17

Figure 8: ResBlock architecture. For the discriminator we removed BN layers in ResBlock.

Table 4: ResNet architectures for CIFAR10 dataset. We use similar architectures to the ones used in Gulrajani et al. (2017).

<table border="1">
<tbody>
<tr>
<td><math>z \in \mathbb{R}^{128} \sim \mathcal{N}(0, I)</math></td>
<td>RGB image <math>x \in \mathbb{R}^{32 \times 32 \times 3}</math></td>
</tr>
<tr>
<td>dense, <math>4 \times 4 \times 256</math></td>
<td>ResBlock down 128</td>
</tr>
<tr>
<td>ResBlock up 256</td>
<td>ResBlock down 128</td>
</tr>
<tr>
<td>ResBlock up 256</td>
<td>ResBlock 128</td>
</tr>
<tr>
<td>ResBlock up 256</td>
<td>ResBlock 128</td>
</tr>
<tr>
<td>BN, ReLU, <math>3 \times 3</math> conv, 3 Tanh</td>
<td>ReLU</td>
</tr>
<tr>
<td></td>
<td>Global sum pooling</td>
</tr>
<tr>
<td>(a) Generator</td>
<td>dense <math>\rightarrow 1</math></td>
</tr>
<tr>
<td></td>
<td>(b) Discriminator</td>
</tr>
</tbody>
</table>

Table 5: ResNet architectures for STL-10 dataset.

<table border="1">
<tbody>
<tr>
<td><math>z \in \mathbb{R}^{128} \sim \mathcal{N}(0, I)</math></td>
<td>RGB image <math>x \in \mathbb{R}^{48 \times 48 \times 3}</math></td>
</tr>
<tr>
<td>dense, <math>6 \times 6 \times 512</math></td>
<td>ResBlock down 64</td>
</tr>
<tr>
<td>ResBlock up 256</td>
<td>ResBlock down 128</td>
</tr>
<tr>
<td>ResBlock up 128</td>
<td>ResBlock down 256</td>
</tr>
<tr>
<td>ResBlock up 64</td>
<td>ResBlock down 512</td>
</tr>
<tr>
<td>BN, ReLU, <math>3 \times 3</math> conv, 3 Tanh</td>
<td>ResBlock 1024</td>
</tr>
<tr>
<td>(a) Generator</td>
<td>ReLU</td>
</tr>
<tr>
<td></td>
<td>Global sum pooling</td>
</tr>
<tr>
<td></td>
<td>dense <math>\rightarrow 1</math></td>
</tr>
<tr>
<td></td>
<td>(b) Discriminator</td>
</tr>
</tbody>
</table>


---

## Page 18

Table 6: ResNet architectures for image generation on ImageNet dataset. For the generator of conditional GANs, we replaced the usual batch normalization layer in the ResBlock with the conditional batch normalization layer. As for the model of the *projection discriminator*, we used the same architecture used in Miyato & Koyama (2018). Please see the paper for the details.

<table border="1">
<tbody>
<tr>
<td><math>z \in \mathbb{R}^{128} \sim \mathcal{N}(0, I)</math></td>
<td>RGB image <math>x \in \mathbb{R}^{128 \times 128 \times 3}</math></td>
<td>RGB image <math>x \in \mathbb{R}^{128 \times 128 \times 3}</math></td>
</tr>
<tr>
<td>dense, <math>4 \times 4 \times 1024</math></td>
<td>ResBlock down 64</td>
<td>ResBlock down 64</td>
</tr>
<tr>
<td>ResBlock up 1024</td>
<td>ResBlock down 128</td>
<td>ResBlock down 128</td>
</tr>
<tr>
<td>ResBlock up 512</td>
<td>ResBlock down 256</td>
<td>ResBlock down 256</td>
</tr>
<tr>
<td>ResBlock up 256</td>
<td>ResBlock down 512</td>
<td>Concat(Embed(<math>y</math>), <math>h</math>)</td>
</tr>
<tr>
<td>ResBlock up 128</td>
<td>ResBlock down 1024</td>
<td>ResBlock down 512</td>
</tr>
<tr>
<td>ResBlock up 64</td>
<td>ResBlock 1024</td>
<td>ResBlock down 1024</td>
</tr>
<tr>
<td>BN, ReLU, <math>3 \times 3</math> conv 3</td>
<td>ReLU</td>
<td>ResBlock 1024</td>
</tr>
<tr>
<td>Tanh</td>
<td>Global sum pooling</td>
<td>ReLU</td>
</tr>
<tr>
<td>(a) Generator</td>
<td>dense <math>\rightarrow 1</math></td>
<td>Global sum pooling</td>
</tr>
<tr>
<td></td>
<td></td>
<td>dense <math>\rightarrow 1</math></td>
</tr>
</tbody>
</table>

(b) Discriminator for unconditional GANs.

(c) Discriminator for conditional GANs. For computational ease, we embedded the integer label  $y \in \{0, \dots, 1000\}$  into 128 dimension before concatenating the vector to the output of the intermediate layer.


---

## Page 19

## C APPENDIX RESULTS

### C.1 ACCURACY OF SPECTRAL NORMALIZATION

Figure 9 shows the spectral norm of each layer in the discriminator over the course of the training. The setting of the optimizer is C in Table 1 throughout the training. In fact, they do not deviate by more than 0.05 for the most part. As an exception, 6 and 7-th convolutional layers with largest rank deviate by more than 0.1 in the beginning of the training, but the norm of this layer too stabilizes around 1 after some iterations.

![Figure 9: A line graph showing the spectral norm of seven convolutional layers (conv0 to conv6) over 100,000 training updates. The y-axis is labeled sigma(W) and ranges from 0.95 to 1.20. The x-axis is labeled 'update' and ranges from 0 to 100,000. All layers start with a spectral norm of 1.0. conv0, conv1, conv2, conv3, and conv4 remain relatively stable around 1.0. conv5 and conv6 show initial spikes above 1.0 (up to ~1.1) before stabilizing back to 1.0.](b86b7fe87a671823a036743e1fd5d232_5_img.webp)

Figure 9: Spectral norms of all seven convolutional layers in the standard CNN during course of the training on CIFAR 10.

### C.2 TRAINING TIME

![Figure 10: Two bar charts comparing the computational time for 100 generator updates across four methods: WGAN-GP, WN, SN, and Vanilla. (a) CIFAR-10 (image size: 32 x 32 x 3) shows times of approximately 40s for WGAN-GP, 30s for WN, 31s for SN, and 28s for Vanilla. (b) STL-10 (image size: 48 x 48 x 3) shows times of approximately 90s for WGAN-GP, 62s for WN, 63s for SN, and 61s for Vanilla.](b86b7fe87a671823a036743e1fd5d232_8_img.webp)

<table border="1">
<caption>Computational time for 100 updates (seconds)</caption>
<thead>
<tr>
<th>Dataset</th>
<th>Method</th>
<th>Time (s)</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="4">CIFAR-10 (32 x 32 x 3)</td>
<td>WGAN-GP</td>
<td>40</td>
</tr>
<tr>
<td>WN</td>
<td>30</td>
</tr>
<tr>
<td>SN</td>
<td>31</td>
</tr>
<tr>
<td>Vanilla</td>
<td>28</td>
</tr>
<tr>
<td rowspan="4">STL-10 (48 x 48 x 3)</td>
<td>WGAN-GP</td>
<td>90</td>
</tr>
<tr>
<td>WN</td>
<td>62</td>
</tr>
<tr>
<td>SN</td>
<td>63</td>
</tr>
<tr>
<td>Vanilla</td>
<td>61</td>
</tr>
</tbody>
</table>

(a) CIFAR-10 (image size:  $32 \times 32 \times 3$ ) (b) STL-10 (image size:  $48 \times 48 \times 3$ )

Figure 10: Computational time for 100 updates. We set  $n_{dis} = 5$

### C.3 THE EFFECT OF $n_{dis}$ ON SPECTRAL NORMALIZATION AND WEIGHT NORMALIZATION

Figure 11 shows the effect of  $n_{dis}$  on the performance of weight normalization and spectral normalization. All results shown in Figure 11 follows setting D, except for the value of  $n_{dis}$ . For WN, the performance deteriorates with larger  $n_{dis}$ , which amounts to computing minimax with better accuracy. Our SN does not suffer from this unintended effect.


---

## Page 20

![Line graph showing Inception scores after 10000 generator updates versus n_dis for SN and WN normalization methods.](885a9a61358566702c5bcaace1ea72e5_2_img.webp)

The graph plots Inception score (y-axis, 2 to 7) against  $n_{dis}$  (x-axis, 12 to 20). Two lines represent SN (red) and WN (blue). Shaded regions indicate variance.

<table border="1">
<caption>Approximate data points from Figure 11</caption>
<thead>
<tr>
<th><math>n_{dis}</math></th>
<th>SN Inception Score</th>
<th>WN Inception Score</th>
</tr>
</thead>
<tbody>
<tr>
<td>12</td>
<td>2.0</td>
<td>5.5</td>
</tr>
<tr>
<td>13</td>
<td>3.5</td>
<td>6.0</td>
</tr>
<tr>
<td>14</td>
<td>5.5</td>
<td>5.8</td>
</tr>
<tr>
<td>15</td>
<td>6.0</td>
<td>5.8</td>
</tr>
<tr>
<td>16</td>
<td>6.3</td>
<td>5.9</td>
</tr>
<tr>
<td>17</td>
<td>6.4</td>
<td>5.8</td>
</tr>
<tr>
<td>18</td>
<td>6.4</td>
<td>5.6</td>
</tr>
<tr>
<td>19</td>
<td>6.4</td>
<td>5.5</td>
</tr>
<tr>
<td>20</td>
<td>6.4</td>
<td>5.4</td>
</tr>
</tbody>
</table>

Figure 11: The effect of  $n_{dis}$  on spectral normalization and weight normalization. The shaded region represents the variance of the result over different seeds.

#### C.4 GENERATED IMAGES ON CIFAR10 WITH GAN-GP, LAYER NORMALIZATION AND BATCH NORMALIZATION

![A grid of generated images on CIFAR-10, comparing GAN-GP, Layer Norm., and Batch Norm. methods across six categories (A-F).](885a9a61358566702c5bcaace1ea72e5_5_img.webp)

The image displays a grid of generated images on the CIFAR-10 dataset. On the left, a small grid labeled 'Dataset' shows the original images. To its right, three rows of generated images are shown, labeled 'GAN-GP', 'Layer Norm.', and 'Batch Norm.'. Each row contains six columns of images, labeled A through F at the bottom. The images in columns A, B, C, and D are real CIFAR-10 images. Column E shows generated images that are highly noisy and lack clear structure. Column F shows generated images that are more coherent and resemble real images.

Figure 12: Generated images with GAN-GP, Layer Norm and Batch Norm on CIFAR-10


---

## Page 21

C.5 IMAGE GENERATION ON IMAGENET
![Figure 13: Learning curves in terms of Inception score for SN-GANs and GANs with orthonormal regularization on ImageNet. (a) Unconditional GANs: SN-GANs (red line with squares) show a higher and more stable Inception score (around 14-15) compared to Orthonormal (blue dashed line with circles) which fluctuates between 11 and 14. (b) Conditional GANs with projection discriminator: SN-GANs (red line with squares) show a much higher and more stable Inception score (around 20-25) compared to Orthonormal (blue dashed line with circles) which fluctuates between 15 and 20. Both plots show iteration on the x-axis (0.0 to 4.5e5) and Inception score on the y-axis.](5d7e6033490dd32e17fff1f9025b1cdd_3_img.webp)

(a) Unconditional GANs

(b) Conditional GANs with *projection discriminator*

Figure 13: Learning curves in terms of Inception score for SN-GANs and GANs with *orthonormal regularization* on ImageNet. The figure (a) shows the results for the standard (unconditional) GANs, and the figure (b) shows the results for the conditional GANs trained with *projection discriminator* (Miyato & Koyama, 2018)

D SPECTRAL NORMALIZATION VS OTHER REGULARIZATION TECHNIQUES

This section is dedicated to the comparative study of spectral normalization and other regularization methods for discriminators. In particular, we will show that contemporary regularizations including weight normalization and weight clipping implicitly impose constraints on weight matrices that places unnecessary restriction on the search space of the discriminator. More specifically, we will show that weight normalization and weight clipping *unwittingly* favor low-rank weight matrices. This can force the trained discriminator to be largely dependent on select few features, rendering the algorithm to be able to match the model distribution with the target distribution only on very low dimensional feature space.

D.1 WEIGHT NORMALIZATION AND FROBENIUS NORMALIZATION

The weight normalization introduced by Salimans & Kingma (2016) is a method that normalizes the  $\ell_2$  norm of each row vector in the weight matrix<sup>8</sup>:

$$\bar{W}_{\text{WN}} := [\bar{w}_1^T, \bar{w}_2^T, \dots, \bar{w}_{d_o}^T]^T, \text{ where } \bar{w}_i(\mathbf{w}_i) := \mathbf{w}_i / \|\mathbf{w}_i\|_2, \quad (25)$$

where  $\bar{w}_i$  and  $\mathbf{w}_i$  are the  $i$ th row vector of  $\bar{W}_{\text{WN}}$  and  $W$ , respectively.

Still another technique to regularize the weight matrix is to use the Frobenius norm:

$$\bar{W}_{\text{FN}} := W / \|W\|_F, \quad (26)$$

where  $\|W\|_F := \sqrt{\text{tr}(W^T W)} = \sqrt{\sum_{i,j} w_{ij}^2}$ .

Originally, these regularization techniques were invented with the goal of improving the generalization performance of supervised training (Salimans & Kingma, 2016; Arpit et al., 2016). However, recent works in the field of GANs (Salimans et al., 2016; Xiang & Li, 2017) found their another *raison d’etat* as a regularizer of discriminators, and succeeded in improving the performance of the original.

<sup>8</sup>In the original literature, the weight normalization was introduced as a method for reparametrization of the form  $\bar{W}_{\text{WN}} := [\gamma_1 \bar{w}_1^T, \gamma_2 \bar{w}_2^T, \dots, \gamma_{d_o} \bar{w}_{d_o}^T]^T$  where  $\gamma_i \in \mathbb{R}$  is to be learned in the course of the training. In this work, we deal with the case  $\gamma_i = 1$  so that we can assess the methods under the Lipschitz constraint.


---

## Page 22

These methods in fact can render the trained discriminator  $D$  to be  $K$ -Lipschitz for a some prescribed  $K$  and achieve the desired effect to a certain extent. However, weight normalization (25) imposes the following implicit restriction on the choice of  $\bar{W}_{\text{WN}}$ :

$$\sigma_1(\bar{W}_{\text{WN}})^2 + \sigma_2(\bar{W}_{\text{WN}})^2 + \cdots + \sigma_T(\bar{W}_{\text{WN}})^2 = d_o, \text{ where } T = \min(d_i, d_o), \quad (27)$$

where  $\sigma_t(A)$  is a  $t$ -th singular value of matrix  $A$ . The above equation holds because  $\sum_{t=1}^{\min(d_i, d_o)} \sigma_t(\bar{W}_{\text{WN}})^2 = \text{tr}(\bar{W}_{\text{WN}}\bar{W}_{\text{WN}}^T) = \sum_{i=1}^{d_o} \frac{\mathbf{w}_i}{\|\mathbf{w}_i\|_2} \frac{\mathbf{w}_i^T}{\|\mathbf{w}_i\|_2} = d_o$ . Under this restriction, the norm  $\|\bar{W}_{\text{WN}}\mathbf{h}\|_2$  for a fixed unit vector  $\mathbf{h}$  is maximized at  $\|\bar{W}_{\text{WN}}\mathbf{h}\|_2 = \sqrt{d_o}$  when  $\sigma_1(\bar{W}_{\text{WN}}) = \sqrt{d_o}$  and  $\sigma_t(\bar{W}_{\text{WN}}) = 0$  for  $t = 2, \dots, T$ , which means that  $\bar{W}_{\text{WN}}$  is of rank one. Using such  $W$  corresponds to using only one feature to discriminate the model probability distribution from the target. Similarly, Frobenius normalization requires  $\sigma_1(\bar{W}_{\text{FN}})^2 + \sigma_2(\bar{W}_{\text{FN}})^2 + \cdots + \sigma_T(\bar{W}_{\text{FN}})^2 = 1$ , and the same argument as above follows.

Here, we see a critical problem in these two regularization methods. In order to retain as much norm of the input as possible and hence to make the discriminator more sensitive, one would hope to make the norm of  $\bar{W}_{\text{WN}}\mathbf{h}$  large. For weight normalization, however, this comes at the cost of reducing the rank and hence the number of features to be used for the discriminator. Thus, there is a conflict of interests between weight normalization and our desire to use as many features as possible to distinguish the generator distribution from the target distribution. The former interest often reigns over the other in many cases, inadvertently diminishing the number of features to be used by the discriminators. Consequently, the algorithm would produce a rather arbitrary model distribution that matches the target distribution only at select few features.

Our spectral normalization, on the other hand, do not suffer from such a conflict in interest. Note that the Lipschitz constant of a linear operator is determined only by the maximum singular value. In other words, the spectral norm is independent of rank. Thus, unlike the weight normalization, our spectral normalization allows the parameter matrix to use as many features as possible while satisfying local 1-Lipschitz constraint. Our spectral normalization leaves more freedom in choosing the number of singular components (features) to feed to the next layer of the discriminator.

To see this more visually, we refer the reader to Figure (14). Note that spectral normalization allows for a wider range of choices than weight normalization.

![Figure 14: Visualization of the difference between spectral normalization (Red) and weight normalization (Blue) on possible sets of singular values. The plot shows the squared singular values (sigma_s^2) on the y-axis (ranging from 0.0 to 1.0) versus the index of s on the x-axis (ranging from 0 to 49). Red curves represent spectral normalization (SN), which are much more spread out and cover a larger area. Blue curves represent weight normalization (WN), which are tightly clustered near the bottom right corner, indicating that the sum of squared singular values is constrained to be 1.](cc0b0aa27c43eb6cd497f339955c017f_8_img.webp)

**Figure 14: Visualization of the difference between spectral normalization (Red) and weight normalization (Blue) on possible sets of singular values.** The possible sets of singular values plotted in increasing order for weight normalization (Blue) and for spectral normalization (Red). For the set of singular values permitted under the spectral normalization condition, we scaled  $\bar{W}_{\text{WN}}$  by  $1/\sqrt{d_o}$  so that its spectral norm is exactly 1. By the definition of the weight normalization, *the area under the blue curves are all bound to be 1*. Note that the range of choice for the weight normalization is small.

In summary, weight normalization and Frobenius normalization favor skewed distributions of singular values, making the column spaces of the weight matrices lie in (approximately) low dimensional vector spaces. On the other hand, our spectral normalization does not compromise the number of feature dimensions used by the discriminator. In fact, we will experimentally show that GANs


---

## Page 23

trained with our spectral normalization can generate a synthetic dataset with wider variety and higher inception score than the GANs trained with other two regularization methods.

## D.2 WEIGHT CLIPPING

Still another regularization technique is *weight clipping* introduced by Arjovsky et al. (2017) in their training of Wasserstein GANs. Weight clipping simply truncates each element of weight matrices so that its absolute value is bounded above by a prescribed constant  $c \in \mathbb{R}_+$ . Unfortunately, weight clipping suffers from the same problem as weight normalization and Frobenius normalization. With weight clipping with the truncation value  $c$ , the value  $\|W\mathbf{x}\|_2$  for a fixed unit vector  $\mathbf{x}$  is maximized when the rank of  $W$  is again one, and the training will again favor the discriminators that use only select few features. Gulrajani et al. (2017) refers to this problem as capacity underuse problem. They also reported that the training of WGAN with weight clipping is slower than that of the original DCGAN (Radford et al., 2016).

## D.3 SINGULAR VALUE CLIPPING AND SINGULAR VALUE CONSTRAINT

One direct and straightforward way of controlling the spectral norm is to clip the singular values (Saito et al., 2017), (Jia et al., 2017). This approach, however, is computationally heavy because one needs to implement singular value decomposition in order to compute all the singular values.

A similar but less obvious approach is to parametrize  $W \in \mathbb{R}^{d_o \times d_i}$  as follows from the get-go and train the discriminators with this constrained parametrization:

$$W := USV^T, \text{ subject to } U^T U = I, V^T V = I, \max_i S_{ii} = K, \quad (28)$$

where  $U \in \mathbb{R}^{d_o \times P}$ ,  $V \in \mathbb{R}^{d_i \times P}$ , and  $S \in \mathbb{R}^{P \times P}$  is a diagonal matrix. However, it is not a simple task to train this model while remaining absolutely faithful to this parametrization constraint. Our spectral normalization, on the other hand, can carry out the updates with relatively low computational cost without compromising the normalization constraint.

## D.4 WGAN WITH GRADIENT PENALTY (WGAN-GP)

Recently, Gulrajani et al. (2017) introduced a technique to enhance the stability of the training of Wasserstein GANs (Arjovsky et al., 2017). In their work, they endeavored to place  $K$ -Lipschitz constraint (5) on the discriminator by augmenting the adversarial loss function with the following regularizer function:

$$\lambda \mathbb{E}_{\hat{\mathbf{x}} \sim p_{\hat{\mathbf{x}}}} [(\|\nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}})\|_2 - 1)^2], \quad (29)$$

where  $\lambda > 0$  is a balancing coefficient and  $\hat{\mathbf{x}}$  is:

$$\hat{\mathbf{x}} := \epsilon \mathbf{x} + (1 - \epsilon) \tilde{\mathbf{x}} \quad (30)$$

$$\text{where } \epsilon \sim U[0, 1], \mathbf{x} \sim p_{\text{data}}, \tilde{\mathbf{x}} = G(\mathbf{z}), \mathbf{z} \sim p_{\mathbf{z}}. \quad (31)$$

Using this augmented objective function, Gulrajani et al. (2017) succeeded in training a GAN based on ResNet (He et al., 2016) with an impressive performance. The advantage of their method in comparison to spectral normalization is that they can impose local 1-Lipschitz constraint directly on the discriminator function without a rather round-about layer-wise normalization. This suggests that their method is less likely to underuse the capacity of the network structure.

At the same time, this type of method that penalizes the gradients at sample points  $\hat{\mathbf{x}}$  suffers from the obvious problem of not being able to regularize the function at the points outside of the support of the current generative distribution. In fact, the generative distribution and its support gradually changes in the course of the training, and this can destabilize the effect of the regularization itself.

On the contrary, our spectral normalization regularizes the function itself, and the effect of the regularization is more stable with respect to the choice of the batch. In fact, we observed in the experiment that a high learning rate can destabilize the performance of WGAN-GP. Training with our spectral normalization does not falter with aggressive learning rate.


---

## Page 24

Moreover, WGAN-GP requires more computational cost than our spectral normalization with single-step power iteration, because the computation of  $\|\nabla_{\mathbf{x}} D\|_2$  requires one whole round of forward and backward propagation. In Figure 10, we compare the computational cost of the two methods for the same number of updates.

Having said that, one shall not rule out the possibility that the gradient penalty can compliment spectral normalization and vice versa. Because these two methods regularizes discriminators by completely different means, and in the experiment section, we actually confirmed that combination of WGAN-GP and reparametrization with spectral normalization improves the quality of the generated examples over the baseline (WGAN-GP only).

## E REPARAMETRIZATION MOTIVATED BY THE SPECTRAL NORMALIZATION

We can take advantage of the regularization effect of the spectral normalization we saw above to develop another algorithm. Let us consider another parametrization of the weight matrix of the discriminator given by:

$$\tilde{W} := \gamma \bar{W}_{\text{SN}} \quad (32)$$

where  $\gamma$  is a scalar variable to be learned. This parametrization compromises the 1-Lipschitz constraint at the layer of interest, but gives more freedom to the model while keeping the model from becoming degenerate. For this reparametrization, we need to control the Lipschitz condition by other means, such as the gradient penalty (Gulrajani et al., 2017). Indeed, we can think of analogous versions of reparametrization by replacing  $\bar{W}_{\text{SN}}$  in (32) with  $W$  normalized by other criterions. The extension of this form is not new. In Salimans & Kingma (2016), they originally introduced weight normalization in order to derive the reparametrization of the form (32) with  $\bar{W}_{\text{SN}}$  replaced (32) by  $W_{\text{WN}}$  and vectorized  $\gamma$ .

### E.1 EXPERIMENTS: COMPARISON OF REPARAMETRIZATION WITH DIFFERENT NORMALIZATION METHODS

In this part of the addendum, we experimentally compare the reparametrizations derived from two different normalization methods (weight normalization and spectral normalization). We tested the reparametrization methods for the training of the discriminator of WGAN-GP. For the architecture of the network in WGAN-GP, we used the same CNN we used in the previous section. For the ResNet-based CNN, we used the same architecture provided by (Gulrajani et al., 2017)<sup>9</sup>.

Tables 7, 8 summarize the result. We see that our method significantly improves the inception score from the baseline on the regular CNN, and slightly improves the score on the ResNet based CNN.

Figure 15 shows the learning curves of (a) critic losses, on train and validation sets and (b) the inception scores with different reparametrization methods. We can see the beneficial effect of spectral normalization in the learning curve of the discriminator as well. We can verify in the figure 15a that the discriminator with spectral normalization overfits less to the training dataset than the discriminator without reparametrization and with weight normalization. The effect of overfitting can be observed on inception score as well, and the final score with spectral normalization is better than the others. As for the best inception score achieved in the course of the training, spectral normalization achieved 7.28, whereas the spectral normalization and vanilla normalization achieved 7.04 and 6.69, respectively.

## F THE GRADIENT OF GENERAL NORMALIZATION METHOD

Let us denote  $\bar{W} := W/N(W)$  to be the normalized weight where  $N(W)$  to be a scalar normalized coefficient (e.g. Spectral norm or Frobenius norm). In general, we can write the derivative of loss

<sup>9</sup>We implement our method based on the open-sourced code provided by the author (Gulrajani et al., 2017) [https://github.com/igul222/improved\\_wgan\\_training/blob/master/gan\\_cifar\\_resnet.py](https://github.com/igul222/improved_wgan_training/blob/master/gan_cifar_resnet.py)


---

## Page 25

<table border="1">
<thead>
<tr>
<th>Method</th>
<th>Inception score</th>
<th>FID</th>
</tr>
</thead>
<tbody>
<tr>
<td>WGAN-GP (Standard CNN, Baseline)</td>
<td>6.68±.06</td>
<td>40.1</td>
</tr>
<tr>
<td>w/ Frobenius Norm.</td>
<td>N/A*</td>
<td>N/A*</td>
</tr>
<tr>
<td>w/ Weight Norm.</td>
<td>6.36±.04</td>
<td>42.4</td>
</tr>
<tr>
<td>w/ Spectral Norm.</td>
<td><b>7.20±.08</b></td>
<td><b>32.0</b></td>
</tr>
<tr>
<td>(WGAN-GP, ResNet, Gulrajani et al. (2017))</td>
<td>7.86±.08</td>
<td></td>
</tr>
<tr>
<td>WGAN-GP (ResNet, Baseline)</td>
<td>7.80±.11</td>
<td>24.5</td>
</tr>
<tr>
<td>w/ Spectral norm.</td>
<td>7.85±.06</td>
<td>23.6</td>
</tr>
<tr>
<td>w/ Spectral norm. (1.5x feature maps in <math>D</math>)</td>
<td><b>7.96±.06</b></td>
<td><b>22.5</b></td>
</tr>
</tbody>
</table>

Table 7: Inception scores with different reparametrization methods on CIFAR10 without label supervisions. (\*)We reported N/A for the inception score and FID of Frobenius normalization because the training collapsed at the early stage.

<table border="1">
<thead>
<tr>
<th>Method (ResNet)</th>
<th>Inception score</th>
<th>FID</th>
</tr>
</thead>
<tbody>
<tr>
<td>(AC-WGAN-GP, Gulrajani et al. (2017))</td>
<td>8.42±.10</td>
<td></td>
</tr>
<tr>
<td>AC-WGAN-GP (Baseline)</td>
<td>8.29±.12</td>
<td>19.5</td>
</tr>
<tr>
<td>w/ Spectral norm.</td>
<td>8.59±.12</td>
<td>18.6</td>
</tr>
<tr>
<td>w/ Spectral norm. (1.5x feature maps in <math>D</math>)</td>
<td><b>8.60±.08</b></td>
<td><b>17.5</b></td>
</tr>
</tbody>
</table>

Table 8: Inception scores and FIDs with different reparametrization methods on CIFAR10 with the label supervision, by auxiliary classifier (Odena et al., 2017).

with respect to unnormalized weight  $W$  as follows:

$$\frac{\partial V(G, D(W))}{\partial W} = \frac{1}{N(W)} \left( \frac{\partial V}{\partial \bar{W}} - \text{trace} \left( \left( \frac{\partial V}{\partial \bar{W}} \right)^T \bar{W} \right) \frac{\partial(N(W))}{\partial W} \right) \quad (33)$$

$$= \frac{1}{N(W)} \left( \nabla_{\bar{W}} V - \text{trace} \left( (\nabla_{\bar{W}} V)^T \bar{W} \right) \nabla_W N \right) \quad (34)$$

$$= \alpha (\nabla_{\bar{W}} V - \lambda \nabla_W N), \quad (35)$$

where  $\alpha := 1/N(W)$  and  $\lambda := \text{trace} \left( (\nabla_{\bar{W}} V)^T \bar{W} \right)$ . The gradient  $\nabla_{\bar{W}} V$  is calculated by  $\hat{\mathbb{E}} [\delta \mathbf{h}^T]$  where  $\delta := (\partial V(G, D)/\partial (\bar{W} \mathbf{h}))^T$ ,  $\mathbf{h}$  is the hidden node in the network to be transformed by  $\bar{W}$  and  $\hat{\mathbb{E}}$  represents empirical expectation over the mini-batch. When  $N(W) := \|W\|_F$ , the derivative is:

$$\frac{\partial V(G, D(W))}{\partial W} = \frac{1}{\|W\|_F} \left( \hat{\mathbb{E}} [\delta \mathbf{h}^T] - \text{trace} \left( \hat{\mathbb{E}} [\delta \mathbf{h}^T]^T \bar{W} \right) \bar{W} \right), \quad (36)$$

and when  $N(W) := \|W\|_2 = \sigma(W)$ ,

$$\frac{\partial V(G, D(W))}{\partial W} = \frac{1}{\sigma(W)} \left( \hat{\mathbb{E}} [\delta \mathbf{h}^T] - \text{trace} \left( \hat{\mathbb{E}} [\delta \mathbf{h}^T]^T \bar{W} \right) \mathbf{u}_1 \mathbf{v}_1^T \right). \quad (37)$$

Notice that, at least for the case  $N(W) := \|W\|_F$  or  $N(W) := \|W\|_2$ , the point of this gradient is given by :

$$\nabla_{\bar{W}} V = k \nabla_W N. \quad (38)$$

where  $\exists k \in \mathbb{R}$


---

## Page 26

![Figure 15: Learning curves of (a) critic loss and (b) inception score on different reparametrization methods on CIFAR-10.](499162dee1d3b084394c5195e604d805_2_img.webp)

Figure 15 consists of two line plots, (a) and (b), showing the performance of different reparametrization methods on CIFAR-10 over 100,000 updates of the generator.

**(a) Critic loss:** This plot shows the critic loss on the y-axis (ranging from -3.0 to 0.0) against the update of the generator on the x-axis (ranging from 0 to 100,000). The legend indicates six series: WGAN-GP (train) (black dashed line), WGAN-GP w/ WN (train) (blue dashed line), WGAN-GP w/ SN (train) (red dashed line), WGAN-GP (valid) (black solid line), WGAN-GP w/ WN (valid) (blue solid line), and WGAN-GP w/ SN (valid) (red solid line). The training curves (dashed) generally show a decrease in loss over time, while the validation curves (solid) show an increase. The WGAN-GP w/ SN method (red) consistently achieves the lowest loss, followed by WGAN-GP w/ WN (blue), and then WGAN-GP (black).

**(b) Inception score:** This plot shows the inception score on the y-axis (ranging from 5.0 to 7.5) against the update of the generator on the x-axis (ranging from 0 to 100,000). The legend indicates three series: WGAN-GP (black solid line), WGAN-GP w/ WN (blue solid line), and WGAN-GP w/ SN (red solid line). All three methods show an increase in the inception score over time, with WGAN-GP w/ SN (red) achieving the highest score, followed by WGAN-GP (black), and then WGAN-GP w/ WN (blue).

Figure 15: Learning curves of (a) critic loss and (b) inception score on different reparametrization method on CIFAR-10 ; weight normalization (WGAN-GP w/ WN), spectral normalization (WGAN-GP w/ SN), and parametrization free (WGAN-GP).
