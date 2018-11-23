import numpy as np


def guess(y, y_, type_size, sess, fd, ctb, th=0.5):
    showy = sess.run(y, feed_dict=fd)
    print(np.argmax(showy,1))
    showy_ = fd[y_]
    h = showy > th
    strict = 0
    lma_p = 0
    lma_r = 0

    for i in range(np.shape(h)[0]):
        if ctb[i] == 1:
            if np.sum(h[i, :]) == 0:
                h[i, np.argmax(showy[i, :])] = 1

            # strict
            count = True
            for j in range(type_size):
                if h[i, j] != showy_[i, j]:
                    count = False
                    break
            if count:
                strict += 1

            # loose macro
            tp = float(np.sum(np.logical_and(h[i], showy_[i])))
            fp = float(np.sum(h[i]))
            tn = float(np.sum(showy_[i]))
            lma_p += tp / fp
            if tn != 0:
                lma_r += tp / tn

    # loose micro
    table = np.transpose(np.tile(ctb, [type_size, 1]))
    true_pos = float(np.sum(np.logical_and(table, np.logical_and(h, showy_))))
    false_pos = float(np.sum(np.logical_and(table, h)))
    true_neg = float(np.sum(np.logical_and(table, showy_)))

    effect = float(np.sum(ctb))
    return (float(strict), lma_p, lma_r, true_pos, false_pos, true_neg, effect)


def test(model, _entity, _context, _label,
         batch_size, sess):
    true_pos = 0
    false_pos = 0
    true_neg = 0
    strict = 0
    lma_p = 0
    lma_r = 0
    effect = 0
    # organization 0 person 13 location 54

    full_size = len(_label)
    for i in range(int(full_size / batch_size)):
        table = np.ones([batch_size])

        fdt = model.fdict(i * batch_size, batch_size, 1,
                          _entity, _context, _label)
        fdt[model.kprob] = 1.0
        result = guess(model.t, model.t_, model.type_size, sess, fdt, table)
        strict += result[0]
        lma_p += result[1]
        lma_r += result[2]
        true_pos += result[3]
        false_pos += result[4]
        true_neg += result[5]
        effect += result[6]
    if false_pos == 0:
        precision = 0
    else:
        precision = true_pos / false_pos
    if true_neg == 0:
        recall = 0
    else:
        recall = true_pos / true_neg
    if effect == 0:
        print('strict: 0')
        print('loose-macro (precision recall f1): %f %f %f' \
              % (0, 0, 0))
    else:
        print('strict: %f' % (strict / effect))
        print('loose-macro (precision recall f1): %f %f %f' \
              % (lma_p / effect, lma_r / effect, (2 * lma_p * lma_r) / (lma_p + lma_r) / effect))
        print('loose-micro (precision recall f1): %f %f %f\n\n' \
              % (precision, recall, (precision * recall * 2) / (precision + recall)))


def dic(w2v, s):
    zeros = []
    for i in range(0, 256):
        zeros.append(0)
    if s in w2v:
        return w2v[s].tolist()
    else:
        return zeros