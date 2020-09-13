def train_encoder_unet(model, dataset, feat_type: FeatureType, show=False):
  class_prototype = None
  lst = []
  
  if show:
    local_start_time = time.time()

  for episode in range(num_episodes):
    selected = np.random.permutation(no_of_samples)[:num_shot + num_query]
    # Create our Support Set.
    support_set = np.array(dataset[selected[:num_shot]])
    # Create our Query Set.
    query_set = np.array(dataset[selected[num_query:]])

    #X_train = support_set[:,0,:]
    X_train = get_cropped_images(support_set, feat_type)
    y_train = support_set[:,1,:]

    X_valid = query_set[:,0,:]
    y_valid = query_set[:,1,:]

    # Get support set features
    support_set_embeddings = encoder(X_train)

    # Get query set features
    query_set_embeddings = encoder(X_valid)

    # Compute loss
    # ¿?

    # Do back propagation
    # ¿?

    # Add support_set_embeddings to a list o whatever to do the mean after loop
    class_prototype = tf.math.reduce_mean(support_set_embeddings[0], axis=0)
    
    lst.append(tf.expand_dims(class_prototype, axis=0))

  # Show execution time
  if show:
    print("--- %s seconds ---" % (time.time() - local_start_time))

  # Do the mean of all class prototypes.
  return tf.reduce_mean(lst, axis=0)
  