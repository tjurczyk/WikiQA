from losses import grad_noise
from lasagne.updates import get_or_compute_grads
import theano
import theano.tensor as T
from lasagne import utils
from collections import OrderedDict
import numpy as np

def gd_adam(loss_or_grads, params, rs, e, noise_eta, noise_decay, learning_rate=0.001, beta1=0.9,
         beta2=0.999, epsilon=1e-8):
    all_grads = get_or_compute_grads(loss_or_grads, params)
    t_prev = theano.shared(utils.floatX(0.))
    updates = OrderedDict()

    t = t_prev + 1
    a_t = learning_rate*T.sqrt(1-beta2**t)/(1-beta1**t)

    for param, g_t in zip(params, all_grads):
        value = param.get_value(borrow=True)
        m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)
        v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)

        m_t = beta1*m_prev + (1-beta1)*g_t
        v_t = beta2*v_prev + (1-beta2)*g_t**2
        step = a_t*m_t/(T.sqrt(v_t) + epsilon)
        noise = T.cast(rs.normal(size=value.shape,avg=0.0,std=noise_eta*step/e**noise_decay),'float32')

        updates[m_prev] = m_t
        updates[v_prev] = v_t
        updates[param] = param - step + noise

    updates[t_prev] = t
    return updates