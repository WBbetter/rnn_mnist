/* ======================================================
 * The Infinite MNIST dataset
 *
 * The files named "data/t10k-*" and "data/train-*" are the original MNIST
 * files (http://yann.lecun.com/exdb/mnist/).  The other files were initially
 * written for the experiments reported in paper "Training Invariant Support
 * Vector Machines using Selective Sampling" by Loosli, Canu, and Bottou
 * (http://leon.bottou.org/papers/loosli-canu-bottou-2006)
 *
 * Copyright (C) 2006- Leon Bottou and Gaelle Loosli                            
 *
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details. You should have received a copy of the GNU General Public
 * License along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111, USA
 *
 * ====================================================== */

#ifndef INFIMNIST_H
#define INFIMNIST_H



/* This library produces an infinite supply of mnist examples by applying
   small deformations and translations. Each example is identified by a long
   integer index that determines the source of the example and the
   transformations applied to the pattern. The examples numbered 0 to 9999 are
   the standard MNIST testing examples. The examples numbered 10000 to 69999
   are the standard MNIST training examples. Each example with indice i
   greater than 69999 is generated by applying a pseudo-random transformation
   to the MNIST training example numbered 10000+((i-10000)%60000). */

typedef struct infimnist_s infimnist_t;

/* Function <infimnist_create> creates the infimnist_t data structure that
   contains the digit data (about 450MB) and caches up to about 1GB worth of
   deformed digit images. The argument <datadir> points to the directory
   containing the data files. Setting it to NULL implicitly selects the
   directory named "data" in the current directory. */

infimnist_t *infimnist_create(const char *datadir);

/* Function <infimnist_destroy> destroys the data structure 
   and returns its memory to the heap. */

void infimnist_destroy(infimnist_t*);


/* Function <infimnist_get_label> returns the label (0 to 9) 
   associated with example <index>. */

int infimnist_get_label(infimnist_t*, long index);

/* Function <infimnist_get_pattern> returns the image associated with the
   example numbered <index>. The image takes the form of a vector of 784
   unsigned bytes organized in row major order.  Each bytes takes a value
   ranging from 0 (white) to 255 (black). There is no need to free the
   resulting pointer as it directly points into the pattern cache. These
   vectors may be automatically deallocated in the future.  However, at any
   time, you can safely access the last ten vectors returned by this
   function. */

const unsigned char *infimnist_get_pattern(infimnist_t*, long index);

/* Function <infimnist_cache_clear> deallocates all the cached patterns
   potentially freeing up to 1GB of memory. */

void infimnist_cache_clear(infimnist_t*);


/* Kernel computation code from (Loosli et al., 2007).
   Added for reference purposes. */

double infimnist_linear_kernel(infimnist_t *p, long i, long j);
double infimnist_rbf_kernel(infimnist_t *p, long i, long j, double gamma);


#endif // INFIMNIST_H

