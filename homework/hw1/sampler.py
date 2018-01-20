import numpy
from numpy import sqrt, cos, sin, log, pi

class ProbabilityModel:

    # Returns a single sample (independent of values returned on previous calls).
    # The returned value is an element of the model's sample space.
    def sample(self):
        pass


# The sample space of this probability model is the set of real numbers, and
# the probability measure is defined by the density function 
# p(x) = 1/(sigma * (2*pi)^(1/2)) * exp(-(x-mu)^2/2*sigma^2)
class UnivariateNormal(ProbabilityModel):
    
    # Initializes a univariate normal probability model object
    # parameterized by mu and (a positive) sigma
    def __init__(self,mu,sigma):
        self.mu = mu
        self.sigma = sigma

    # Marsaglia polar method
    # def sample(self):
    #     u = numpy.random.uniform(-1, 1);
    #     v = numpy.random.uniform(-1, 1);
    #     s = numpy.square(u) + numpy.square(v);
    #     res = u * sqrt(-2 * log(s) / s);
    #     return self.mu + self.sigma * s;
    
    # Box-Muller transformation    
    def sample(self):
        u = numpy.random.uniform(0, 1);
        v = numpy.random.uniform(0, 1);
        z = sqrt(-2 * log(u)) * cos(2 * pi * v);
        return self.mu + self.sigma * z;

    
# The sample space of this probability model is the set of D dimensional real
# column vectors (modeled as numpy.array of size D x 1), and the probability 
# measure is defined by the density function 
# p(x) = 1/(det(Sigma)^(1/2) * (2*pi)^(D/2)) * exp( -(1/2) * (x-mu)^T * Sigma^-1 * (x-mu) )
class MultiVariateNormal(ProbabilityModel):
    
    # Initializes a multivariate normal probability model object 
    # parameterized by Mu (numpy.array of size D x 1) expectation vector 
    # and symmetric positive definite covariance Sigma (numpy.array of size D x D)
    def __init__(self,Mu,Sigma):
        self.Mu = Mu;
        self.Sigma = Sigma;

    def sample(self):
        Z = [];
        # Cholesky decomposition
        A = numpy.linalg.cholesky(self.Sigma);
        for i in range(0, self.Mu.size):
            Z.append(UnivariateNormal(0, 1).sample());
        Z = numpy.array(Z);
        return self.Mu + numpy.dot(A, Z)
    

# The sample space of this probability model is the finite discrete set {0..k-1}, and 
# the probability measure is defined by the atomic probabilities 
# P(i) = ap[i]
class Categorical(ProbabilityModel):
    
    # Initializes a categorical (a.k.a. multinom, multinoulli, finite discrete) 
    # probability model object with distribution parameterized by the atomic probabilities vector
    # ap (numpy.array of size k).
    def __init__(self,ap):
        self.ap = ap;

    def sample(self):
        z = numpy.array(self.ap);
        return numpy.sum(numpy.cumsum(z) < numpy.random.uniform(0, 1));


# The sample space of this probability model is the union of the sample spaces of 
# the underlying probability models, and the probability measure is defined by 
# the atomic probability vector and the densities of the supplied probability models
# p(x) = sum ad[i] p_i(x)
class MixtureModel(ProbabilityModel):
    
    # Initializes a mixture-model object parameterized by the
    # atomic probabilities vector ap (numpy.array of size k) and by the tuple of 
    # probability models pm
    def __init__(self,ap,pm):
        self.ap = ap;
        self.pm = pm;

    def sample(self):
        z = numpy.array(self.ap);
        return self.pm[numpy.sum(numpy.cumsum(z) < numpy.random.uniform(0, 1))].sample();

