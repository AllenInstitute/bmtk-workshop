import tensorflow as tf


# import tensorflow.compat.v2 as tf

# from keras.optimizers import optimizer
# from keras.saving.object_registration import register_keras_serializable

# # Same TF2.15 / Keras 2.12 public export for consistency
# from tensorflow.python.util.tf_export import keras_export


class ExponentiatedAdam(tf.keras.optimizers.Optimizer):
    r"""Adam-like optimizer with *exponentiated* gradient updates.

    This rewrites the final update from

        w <- w - alpha * m / (sqrt(v) + epsilon)

    to

        w <- w * exp(- alpha * m / (sqrt(v) + epsilon) * sign(w)),

    preserving the rest of the Adam algorithm (moments `m`, `v`, AMSGrad, etc.).
    By default, `sign(0) = +1` so zero-valued parameters can still move off zero.

    Sparse updates are applied consistently via `scatter_mul()` on the affected
    slices only.

    Reference:
      - "Brain-like learning with exponentiated gradients" (and related papers).
      - The original Adam reference:
        [Kingma et al., 2014](http://arxiv.org/abs/1412.6980).
    """

    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        jit_compile=True,
        name="ExponentiatedAdam",
        **kwargs
    ):
        """Create a new ExponentiatedAdam optimizer."""
        super().__init__(
            name=name,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            jit_compile=jit_compile,
            **kwargs
        )
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.amsgrad = amsgrad

    def build(self, var_list):
        """Initialize optimizer variables.

        Similar to Adam: we have slot variables for
        - m (first moment)
        - v (second moment)
        - vhat (if amsgrad=True, for the max of second moments).
        """
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self._built = True
        self._momentums = []
        self._velocities = []
        for var in var_list:
            self._momentums.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="m"
                )
            )
            self._velocities.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="v"
                )
            )
        if self.amsgrad:
            self._velocity_hats = []
            for var in var_list:
                self._velocity_hats.append(
                    self.add_variable_from_reference(
                        model_variable=var, variable_name="vhat"
                    )
                )

    @tf.function(jit_compile=True)
    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""
        # Get current iteration
        local_step = tf.cast(self.iterations + 1, variable.dtype)
        # Compute powers of beta_1 and beta_2
        beta_1_t = tf.cast(self.beta_1, variable.dtype)
        beta_2_t = tf.cast(self.beta_2, variable.dtype)
        beta_1_power = tf.pow(beta_1_t, local_step)
        beta_2_power = tf.pow(beta_2_t, local_step)

        # Get the current learning rate (supports schedules)
        lr = tf.cast(self.learning_rate, variable.dtype)
        # Standard Adam alpha correction
        alpha = lr * tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)

        # Fetch the slot variables for this parameter
        var_key = self._var_key(variable)
        m = self._momentums[self._index_dict[var_key]]
        v = self._velocities[self._index_dict[var_key]]

        # ------------------------
        # Sparse vs. Dense branch
        # ------------------------
        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradient
            # 1) Update m (first moment)
            m.assign_add(-m * (1 - beta_1_t))  # Decay existing m
            m.scatter_add(
                tf.IndexedSlices(
                    gradient.values * (1 - beta_1_t),
                    gradient.indices,
                )
            )
            # 2) Update v (second moment)
            v.assign_add(-v * (1 - beta_2_t))
            v.scatter_add(
                tf.IndexedSlices(
                    tf.square(gradient.values) * (1 - beta_2_t),
                    gradient.indices,
                )
            )
            # 3) AMSGrad if needed
            if self.amsgrad:
                vhat = self._velocity_hats[self._index_dict[var_key]]
                vhat.assign(tf.maximum(vhat, v))
                v_used = vhat
            else:
                v_used = v

            # 4) Exponentiated update for only the slices that changed
            # Gather the relevant slices for var, m, v
            var_slices = tf.gather(variable, gradient.indices)
            m_slices = tf.gather(m, gradient.indices)
            v_slices = tf.gather(v_used, gradient.indices)

            # adam_grad = m / (sqrt(v)+eps)
            adam_grad_slices = m_slices / (tf.sqrt(v_slices) + self.epsilon)

            # sign(w) for these slices, fallback sign(0)=+1
            sign_w_slices = tf.sign(var_slices)
            sign_w_slices = tf.where(
                tf.equal(sign_w_slices, 0), tf.ones_like(sign_w_slices), sign_w_slices
            )

            # exponent = - alpha * adam_grad_slices * sign_w_slices
            exponent_slices = -alpha * adam_grad_slices * sign_w_slices

            # multiplier = exp(exponent_slices)
            multiplier_slices = tf.exp(exponent_slices)

            # var_slices_new = var_slices * multiplier_slices
            # We can do partial update with scatter_mul:
            #   new_var[i] = old_var[i] * multiplier[i]
            variable.scatter_mul(
                tf.IndexedSlices(multiplier_slices, gradient.indices)
            )

        else:
            # Dense gradient
            # 1) Update m
            m.assign_add((gradient - m) * (1 - beta_1_t))
            # 2) Update v
            v.assign_add((tf.square(gradient) - v) * (1 - beta_2_t))
            # 3) AMSGrad
            if self.amsgrad:
                vhat = self._velocity_hats[self._index_dict[var_key]]
                vhat.assign(tf.maximum(vhat, v))
                v_used = vhat
            else:
                v_used = v

            # 4) Exponentiated update
            adam_grad = m / (tf.sqrt(v_used) + self.epsilon)

            # sign(w), fallback sign(0)=+1
            sign_w = tf.sign(variable)
            sign_w = tf.where(tf.equal(sign_w, 0), tf.ones_like(sign_w), sign_w)

            exponent = -alpha * adam_grad * sign_w
            multiplier = tf.exp(exponent)
            variable.assign(variable * multiplier)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    self._learning_rate
                ),
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
            }
        )
        return config


# This may work with Keras 3.3.0, but it's not guaranteed.
# class ExponentiatedAdam(tf.keras.optimizers.Optimizer):
#     """
#     Adam-like optimizer with exponentiated gradient updates,
#     incorporating the sign of w into the exponent.
#     """
#     def __init__(
#         self,
#         learning_rate=0.001,
#         beta_1=0.9,
#         beta_2=0.999,
#         epsilon=1e-7,
#         amsgrad=False,
#         name="ExponentiatedAdam",
#         **kwargs
#     ):
#         super().__init__(name, **kwargs)
#         # self._set_hyper("learning_rate", learning_rate)
#         # self._set_hyper("beta_1", beta_1)
#         # self._set_hyper("beta_2", beta_2)
#         self.epsilon = epsilon
#         self.amsgrad = amsgrad

#     def _create_slots(self, var_list):
#         # Create Adam-like slot variables for first and second moments.
#         for var in var_list:
#             self.add_slot(var, "m")  # momentum
#             self.add_slot(var, "v")  # velocity
#             if self.amsgrad:
#                 self.add_slot(var, "vhat")  # for AMSGrad

#     @tf.function
#     def _resource_apply_dense(self, grad, var):
#         # Gather hyperparameters
#         var_dtype = var.dtype.base_dtype
#         lr_t = self._decayed_lr(var_dtype)  # handles learning rate schedules
#         beta_1_t = self._get_hyper("beta_1", var_dtype)
#         beta_2_t = self._get_hyper("beta_2", var_dtype)

#         local_step = tf.cast(self.iterations + 1, var_dtype)
#         beta_1_power = tf.pow(beta_1_t, local_step)
#         beta_2_power = tf.pow(beta_2_t, local_step)

#         # Fetch existing slot variables
#         m = self.get_slot(var, "m")
#         v = self.get_slot(var, "v")

#         # Update first moment estimate
#         m_t = m.assign(
#             beta_1_t * m + (1.0 - beta_1_t) * grad,
#             use_locking=self._use_locking
#         )
#         # Update second moment estimate
#         v_t = v.assign(
#             beta_2_t * v + (1.0 - beta_2_t) * tf.square(grad),
#             use_locking=self._use_locking
#         )

#         # Possibly AMSGrad
#         if self.amsgrad:
#             vhat = self.get_slot(var, "vhat")
#             vhat_t = vhat.assign(tf.maximum(vhat, v_t), use_locking=self._use_locking)
#             v_used = vhat_t
#         else:
#             v_used = v_t

#         # Compute the "Adam gradient" = (m / (sqrt(v) + eps)) with bias correction:
#         # alpha = lr_t * sqrt(1 - beta_2_power) / (1 - beta_1_power)
#         alpha_t = lr_t * tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)
#         adam_grad = m_t / (tf.sqrt(v_used) + self.epsilon)
        
#         # Now incorporate sign(w). sign(0) -> +1 to avoid 0 being stuck
#         sign_w = tf.sign(var)
#         sign_w = tf.where(tf.equal(sign_w, 0), tf.ones_like(sign_w), sign_w)

#         # Exponent = -alpha_t * adam_grad * sign(w)
#         exponent = -alpha_t * adam_grad * sign_w

#         # Exponentiated update: w <- w * exp(exponent)
#         var_update = var.assign(var * tf.exp(exponent), use_locking=self._use_locking)

#         return tf.group(var_update, m_t, v_t)

#     def _resource_apply_sparse(self, grad, var, indices):
#         # If you have embeddings/sparse updates, implement similarly
#         # or raise NotImplementedError
#         raise NotImplementedError("Sparse gradient updates are not implemented.")

#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "learning_rate": self._serialize_hyperparameter("learning_rate"),
#             "beta_1": self._serialize_hyperparameter("beta_1"),
#             "beta_2": self._serialize_hyperparameter("beta_2"),
#             "epsilon": self.epsilon,
#             "amsgrad": self.amsgrad,
#         })
#         return config
