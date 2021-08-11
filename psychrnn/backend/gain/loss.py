import tensorflow as tf

def rt_mask_mse_06(predictions, y, output_mask):
    
    # find decision period
    output1 = tf.equal(y, 1)
    decision_period_vec = tf.expand_dims(tf.reduce_any(output1, axis=2), axis=2)

    # find predictions greater than threshold
    above_thresh = tf.greater(predictions, 0.6)
    above_thresh_vec = tf.reduce_any(above_thresh, axis=2)
    above_thresh_cum = tf.cumsum(tf.cast(above_thresh_vec, dtype=tf.float32), axis=1)
    above_thresh_cont = tf.expand_dims(tf.greater_equal(above_thresh_cum, 1), axis=2)

    # get union of decision period and predictions greater than threshold
    decision_thresh = tf.concat((decision_period_vec, above_thresh_cont), axis=2)
    decision_mask_vec = tf.expand_dims(tf.reduce_all(decision_thresh, axis=2), axis=2)
    decision_mask = decision_mask_vec
    for i in range(output_mask.shape[2]-1):
        decision_mask = tf.concat((decision_mask, decision_mask_vec), axis=2)
    
    # adjust output matrix
    baseline_y_mat = tf.fill(decision_mask.shape, 0.2)
    y_reshape = tf.reshape(y, decision_mask.shape)
    y_decision_mask = tf.where(decision_mask, baseline_y_mat, y_reshape)

    return tf.reduce_mean(input_tensor=tf.square(output_mask * (predictions - y_decision_mask)))

    # # apply output_mask
    # out_mask_bool = tf.equal(output_mask, 1)
    # pred_out_mask = tf.boolean_mask(predictions, out_mask_bool)
    # y_out_mask = tf.boolean_mask(y_reshape, y_decision_mask)
  
    # return tf.reduce_mean(input_tensor=tf.square(pred_out_mask - y_out_mask))

def rt_mask_mse_07(predictions, y, output_mask):
    
    # find decision period
    output1 = tf.equal(y, 1)
    # define whether decision has been made: 1: yes; 0: not yet
    decision_period_vec = tf.expand_dims(tf.reduce_any(output1, axis=2), axis=2)

    # find predictions greater than threshold
    above_thresh = tf.greater(predictions, 0.7)
    above_thresh_vec = tf.reduce_any(above_thresh, axis=2)
    above_thresh_cum = tf.cumsum(tf.cast(above_thresh_vec, dtype=tf.float32), axis=1)
    above_thresh_cont = tf.expand_dims(tf.greater_equal(above_thresh_cum, 1), axis=2)

    # get union of decision period and predictions greater than threshold
    decision_thresh = tf.concat((decision_period_vec, above_thresh_cont), axis=2)
    decision_mask_vec = tf.expand_dims(tf.reduce_all(decision_thresh, axis=2), axis=2)
    decision_mask = decision_mask_vec
    for i in range(output_mask.shape[2]-1):
        decision_mask = tf.concat((decision_mask, decision_mask_vec), axis=2)
    
    # adjust output matrix
    baseline_y_mat = tf.fill(decision_mask.shape, 0.2)
    y_reshape = tf.reshape(y, decision_mask.shape)
    y_decision_mask = tf.where(decision_mask, baseline_y_mat, y_reshape)

    return tf.reduce_mean(input_tensor=tf.square(output_mask * (predictions - y_decision_mask)))

    # # apply output_mask
    # out_mask_bool = tf.equal(output_mask, 1)
    # pred_out_mask = tf.boolean_mask(predictions, out_mask_bool)
    # y_out_mask = tf.boolean_mask(y_reshape, y_decision_mask)
  
    # return tf.reduce_mean(input_tensor=tf.square(pred_out_mask - y_out_mask))

def rt_mask_mse_08(predictions, y, output_mask):
    
    # find decision period
    output1 = tf.equal(y, 1)
    decision_period_vec = tf.expand_dims(tf.reduce_any(output1, axis=2), axis=2)

    # find predictions greater than threshold
    above_thresh = tf.greater(predictions, 0.8)
    above_thresh_vec = tf.reduce_any(above_thresh, axis=2)
    above_thresh_cum = tf.cumsum(tf.cast(above_thresh_vec, dtype=tf.float32), axis=1)
    above_thresh_cont = tf.expand_dims(tf.greater_equal(above_thresh_cum, 1), axis=2)

    # get union of decision period and predictions greater than threshold
    decision_thresh = tf.concat((decision_period_vec, above_thresh_cont), axis=2)
    decision_mask_vec = tf.expand_dims(tf.reduce_all(decision_thresh, axis=2), axis=2)
    decision_mask = decision_mask_vec
    for i in range(output_mask.shape[2]-1):
        decision_mask = tf.concat((decision_mask, decision_mask_vec), axis=2)
    
    # adjust output matrix
    baseline_y_mat = tf.fill(decision_mask.shape, 0.2)
    y_reshape = tf.reshape(y, decision_mask.shape)
    y_decision_mask = tf.where(decision_mask, baseline_y_mat, y_reshape)

    return tf.reduce_mean(input_tensor=tf.square(output_mask * (predictions - y_decision_mask)))

    # # apply output_mask
    # out_mask_bool = tf.equal(output_mask, 1)
    # pred_out_mask = tf.boolean_mask(predictions, out_mask_bool)
    # y_out_mask = tf.boolean_mask(y_reshape, y_decision_mask)
  
    # return tf.reduce_mean(input_tensor=tf.square(pred_out_mask - y_out_mask))