# Vulnerable/patched pairs with and without the bridge

469 test pairs (first vulnerable and first benign function of each commit). Same-function pairs are those where both members define the same function; the examples below are drawn from them. Outcomes are at each model's validation-calibrated threshold. 'fixes' = correct with the bridge but not without it; 'breaks' = the reverse.


## codebert_full: same-function pairs 91: correct without bridge 1, with bridge 2; fixes 2, breaks 1, both fail 88


All 469 pairs: P-C 18.55% without vs 14.50% with the bridge | fixes 20, breaks 39, both fail 362


### fixes (0 short; top 2)

- **tensorflow**  ['CWE-476'] commit 9e82dce6e6 | 90 lines, 9 changed | without bridge 0.107/0.120 (thr 0.05), with bridge 0.868/0.747 (thr 0.77) | KG slots 32/32, same entities: True | features changed: {'loc': (90.0, 97.0), 'halstead_n1': (16.0, 17.0), 'halstead_n2': (130.0, 133.0), 'halstead_N1': (309.0, 330.0), 'halstead_N2': (299.0, 322.0), 'halstead_vol': (4371.41, 4713.19)}

```diff
@@ -10,3 +10,3 @@
             "Input 0 (file_pattern) must be a string scalar; got a tensor of ",
-            size, "elements"));
+            size, " elements"));
   }
@@ -15,2 +15,9 @@
   const Tensor& tensor_name_t = context->input(1);
+  {
+    const int64_t size = tensor_name_t.NumElements();
+    OP_REQUIRES(context, size > restore_index,
+                errors::InvalidArgument(
+                    "Input 1 (file_pattern) must be a have at least ",
+                    restore_index + 1, " elements"));
+  }
   const string& tensor_name = tensor_name_t.flat<tstring>()(restore_index);
```

- **tensorflow**  ['CWE-125'] commit 002408c369 | 141 lines, 232 changed | without bridge 0.141/0.191 (thr 0.05), with bridge 0.908/0.286 (thr 0.77) | KG slots 32/32, same entities: False | features changed: {'loc': (141.0, 119.0), 'cyclomatic': (8.0, 9.0), 'max_depth': (7.0, 6.0), 'n_arrays': (0.0, 38.0), 'n_ptr_arith': (2.0, 0.0), 'n_null_check': (4.0, 0.0), 'halstead_n1': (24.0, 25.0), 'halstead_n2': (200.0, 126.0), 'halstead_N1': (560.0, 547.0), 'halstead_N2': (637.0, 448.0), 'halstead_vol': (9345.4, 7202.21), 'fan_out': (29.0, 30.0)}

```diff
@@ -1,125 +1,112 @@
 void Compute(OpKernelContext* context) override {
-    // Here's the basic idea:
-    // Batch and depth dimension are independent from row and col dimension. And
-    // because FractionalAvgPool currently only support pooling along row and
-    // col, we can basically think of this 4D tensor backpropagation as
-    // operation of a series of 2D planes.
-    //
-    // For each element of a 'slice' (2D plane) of output_backprop, we need to
-    // figure out its contributors when doing FractionalAvgPool operation. This
-    // can be done based on row_pooling_sequence, col_pooling_seq and
-    // overlapping.
-    // Once we figure out the original contributors, we just need to evenly
-    // divide the value of this element among these contributors.
-    //
-    // Internally, we divide the out_backprop tensor and store it in a temporary
-    // tensor of double type. And cast it to the corresponding type.
     typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
         ConstEigenMatrixMap;
-    typedef Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
-        EigenDoubleMatrixMap;
-
-    // Grab the inputs.
-    const Tensor& orig_input_tensor_shape = context->input(0);
-    OP_REQUIRES(context,
-                orig_input_tensor_shape.dims() == 1 &&
-                    orig_input_tensor_shape.NumElements() == 4,
-                errors::InvalidArgument("original input tensor shape must be"
-                                        "1-dimensional and 4 elements"));
-    const Tensor& out_backprop = context->input(1);
-    const Tensor& row_seq_tensor = context->input(2);
-    const Tensor& col_seq_tensor = context->input(3);
-
-    const int64_t out_batch = out_backprop.dim_size(0);
-    const int64_t out_rows = out_backprop.dim_size(1);
-    const int64_t out_cols = out_backprop.dim_size(2);
-    const int64_t out_depth = out_backprop.dim_size(3);
-
```


### breaks (0 short; top 1)

- **pjproject**  ['CWE-787'] commit 450baca94f | 65 lines, 65 changed | without bridge 0.058/0.024 (thr 0.05), with bridge 0.471/0.005 (thr 0.77) | KG slots 32/5, same entities: False | features changed: {'loc': (65.0, 14.0), 'cyclomatic': (10.0, 2.0), 'max_depth': (3.0, 2.0), 'n_returns': (4.0, 2.0), 'n_derefs': (13.0, 3.0), 'n_size_t': (1.0, 0.0), 'halstead_n1': (23.0, 14.0), 'halstead_n2': (64.0, 19.0), 'halstead_N1': (197.0, 37.0), 'halstead_N2': (163.0, 31.0), 'halstead_vol': (2319.46, 343.02), 'fan_out': (6.0, 4.0), 'n_signed_cmp': (1.0, 0.0)}

```diff
@@ -1,6 +1,4 @@
-PJ_DEF(pj_status_t) pjstun_parse_msg( void *buf, pj_size_t buf_len, 
-				      pjstun_msg *msg)
+PJ_DEF(void*) pjstun_msg_find_attr( pjstun_msg *msg, pjstun_attr_type t)
 {
-    pj_uint16_t msg_type, msg_len;
-    char *p_attr;
+    int i;
 
@@ -8,58 +6,9 @@
 
-    msg->hdr = (pjstun_msg_hdr*)buf;
-    msg_type = pj_ntohs(msg->hdr->type);
-
-    switch (msg_type) {
-    case PJSTUN_BINDING_REQUEST:
-    case PJSTUN_BINDING_RESPONSE:
-    case PJSTUN_BINDING_ERROR_RESPONSE:
-    case PJSTUN_SHARED_SECRET_REQUEST:
-    case PJSTUN_SHARED_SECRET_RESPONSE:
-    case PJSTUN_SHARED_SECRET_ERROR_RESPONSE:
-	break;
-    default:
-	PJ_LOG(4,(THIS_FILE, "Error: unknown msg type %d", msg_type));
-	return PJLIB_UTIL_ESTUNINMSGTYPE;
+    for (i=0; i<msg->attr_count; ++i) {
+	pjstun_attr_hdr *attr = msg->attr[i];
+	if (pj_ntohs(attr->type) == t)
+	    return attr;
     }
 
-    msg_len = pj_ntohs(msg->hdr->length);
-    if (msg_len != buf_len - sizeof(pjstun_msg_hdr)) {
-	PJ_LOG(4,(THIS_FILE, "Error: invalid msg_len %d (expecting %d)", 
-			     msg_len, buf_len - sizeof(pjstun_msg_hdr)));
-	return PJLIB_UTIL_ESTUNINMSGLEN;
-    }
-
```


### both_fail (12 short; top 5)

- **linux**  ['CWE-834'] commit 775c5033a0 | 4 lines, 1 changed | without bridge 0.022/0.022 (thr 0.05), with bridge 0.003/0.004 (thr 0.77) | KG slots 32/32, same entities: False | features changed: {'loc': (4.0, 5.0), 'halstead_n2': (10.0, 11.0), 'halstead_N1': (13.0, 16.0), 'halstead_N2': (12.0, 14.0), 'halstead_vol': (106.2, 129.66), 'fan_out': (3.0, 4.0)}

```diff
@@ -2,2 +2,3 @@
 {
+	remove_inode_hash(inode);
 	set_bit(FUSE_I_BAD, &get_fuse_inode(inode)->state);
```

- **tensorflow**  ['CWE-125'] commit 701cfaca22 | 8 lines, 2 changed | without bridge 0.029/0.043 (thr 0.05), with bridge 0.027/0.425 (thr 0.77) | KG slots 8/10, same entities: False | features changed: {'loc': (8.0, 10.0), 'n_derefs': (9.0, 11.0), 'halstead_n1': (11.0, 12.0), 'halstead_n2': (22.0, 26.0), 'halstead_N1': (55.0, 68.0), 'halstead_N2': (44.0, 54.0), 'halstead_vol': (499.4, 640.25), 'fan_out': (8.0, 10.0)}

```diff
@@ -1,2 +1,4 @@
 Status SparseCountSparseOutputShapeFn(InferenceContext *c) {
+  ShapeHandle unused;
+  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &unused));
   auto rank = c->Dim(c->input(0), 1);
```

- **qemu**  ['CWE-835'] commit 1caff0340f | 14 lines, 2 changed | without bridge 0.026/0.025 (thr 0.05), with bridge 0.007/0.007 (thr 0.77) | KG slots 32/32, same entities: False | features changed: {'n_derefs': (5.0, 3.0), 'halstead_n2': (32.0, 31.0), 'halstead_N1': (57.0, 55.0), 'halstead_N2': (48.0, 46.0), 'halstead_vol': (573.24, 548.05)}

```diff
@@ -7,3 +7,3 @@
     if (s->phy_reg[PHY_CTRL] & MII_CR_LOOPBACK) {
-        nc->info->receive(nc, buf, size);
+        qemu_receive_packet(nc, buf, size);
     } else {
```

- **vim**  ['CWE-476'] commit 8e4b76da1d | 26 lines, 2 changed | without bridge 0.023/0.023 (thr 0.05), with bridge 0.005/0.005 (thr 0.77) | KG slots 4/4, same entities: True | features changed: {'cyclomatic': (5.0, 6.0), 'n_derefs': (1.0, 2.0), 'n_null_check': (2.0, 3.0), 'halstead_n2': (41.0, 42.0), 'halstead_N1': (65.0, 69.0), 'halstead_N2': (70.0, 74.0), 'halstead_vol': (780.48, 830.45)}

```diff
@@ -14,3 +14,3 @@
 	    match = name;
-	else
+	else if (rmp->regprog != NULL)
 	{
```

- **nagioscore**  ['CWE-476'] commit b1a92a3b52 | 33 lines, 2 changed | without bridge 0.022/0.022 (thr 0.05), with bridge 0.007/0.007 (thr 0.77) | KG slots 16/16, same entities: True | features changed: {'n_null_check': (2.0, 3.0), 'halstead_N1': (103.0, 102.0), 'halstead_N2': (106.0, 107.0)}

```diff
@@ -4,3 +4,3 @@
 
-	if (!*buf || !strcmp(buf, "help")) {
+	if (buf == NULL || !strcmp(buf, "help")) {
 		nsock_printf_nul(sd,
```


## unixcoder_full: same-function pairs 91: correct without bridge 1, with bridge 0; fixes 0, breaks 1, both fail 90


All 469 pairs: P-C 13.43% without vs 12.15% with the bridge | fixes 8, breaks 14, both fail 398


### fixes (0 short; top 0)


### breaks (0 short; top 1)

- **dpdk**  ['CWE-703'] commit af74f7db38 | 96 lines, 3 changed | without bridge 0.640/0.038 (thr 0.04), with bridge 0.913/0.849 (thr 0.12) | KG slots 32/32, same entities: False | features changed: {'loc': (96.0, 99.0), 'cyclomatic': (11.0, 12.0), 'n_returns': (4.0, 5.0), 'n_null_check': (1.0, 2.0), 'halstead_n2': (100.0, 101.0), 'halstead_N1': (334.0, 342.0), 'halstead_N2': (331.0, 339.0), 'halstead_vol': (4632.25, 4751.53), 'fan_out': (10.0, 11.0)}

```diff
@@ -12,2 +12,5 @@
 	int numa_node = SOCKET_ID_ANY;
+
+	if (validate_msg_fds(dev, ctx, 1) != 0)
+		return RTE_VHOST_MSG_RESULT_ERR;
 
```


### both_fail (12 short; top 5)

- **linux**  ['CWE-834'] commit 775c5033a0 | 4 lines, 1 changed | without bridge 0.000/0.000 (thr 0.04), with bridge 0.001/0.001 (thr 0.12) | KG slots 32/32, same entities: False | features changed: {'loc': (4.0, 5.0), 'halstead_n2': (10.0, 11.0), 'halstead_N1': (13.0, 16.0), 'halstead_N2': (12.0, 14.0), 'halstead_vol': (106.2, 129.66), 'fan_out': (3.0, 4.0)}

```diff
@@ -2,2 +2,3 @@
 {
+	remove_inode_hash(inode);
 	set_bit(FUSE_I_BAD, &get_fuse_inode(inode)->state);
```

- **tensorflow**  ['CWE-125'] commit 701cfaca22 | 8 lines, 2 changed | without bridge 0.002/0.012 (thr 0.04), with bridge 0.001/0.009 (thr 0.12) | KG slots 8/10, same entities: False | features changed: {'loc': (8.0, 10.0), 'n_derefs': (9.0, 11.0), 'halstead_n1': (11.0, 12.0), 'halstead_n2': (22.0, 26.0), 'halstead_N1': (55.0, 68.0), 'halstead_N2': (44.0, 54.0), 'halstead_vol': (499.4, 640.25), 'fan_out': (8.0, 10.0)}

```diff
@@ -1,2 +1,4 @@
 Status SparseCountSparseOutputShapeFn(InferenceContext *c) {
+  ShapeHandle unused;
+  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &unused));
   auto rank = c->Dim(c->input(0), 1);
```

- **qemu**  ['CWE-835'] commit 1caff0340f | 14 lines, 2 changed | without bridge 0.001/0.000 (thr 0.04), with bridge 0.001/0.001 (thr 0.12) | KG slots 32/32, same entities: False | features changed: {'n_derefs': (5.0, 3.0), 'halstead_n2': (32.0, 31.0), 'halstead_N1': (57.0, 55.0), 'halstead_N2': (48.0, 46.0), 'halstead_vol': (573.24, 548.05)}

```diff
@@ -7,3 +7,3 @@
     if (s->phy_reg[PHY_CTRL] & MII_CR_LOOPBACK) {
-        nc->info->receive(nc, buf, size);
+        qemu_receive_packet(nc, buf, size);
     } else {
```

- **vim**  ['CWE-476'] commit 8e4b76da1d | 26 lines, 2 changed | without bridge 0.001/0.001 (thr 0.04), with bridge 0.001/0.002 (thr 0.12) | KG slots 4/4, same entities: True | features changed: {'cyclomatic': (5.0, 6.0), 'n_derefs': (1.0, 2.0), 'n_null_check': (2.0, 3.0), 'halstead_n2': (41.0, 42.0), 'halstead_N1': (65.0, 69.0), 'halstead_N2': (70.0, 74.0), 'halstead_vol': (780.48, 830.45)}

```diff
@@ -14,3 +14,3 @@
 	    match = name;
-	else
+	else if (rmp->regprog != NULL)
 	{
```

- **nagioscore**  ['CWE-476'] commit b1a92a3b52 | 33 lines, 2 changed | without bridge 0.000/0.001 (thr 0.04), with bridge 0.001/0.001 (thr 0.12) | KG slots 16/16, same entities: True | features changed: {'n_null_check': (2.0, 3.0), 'halstead_N1': (103.0, 102.0), 'halstead_N2': (106.0, 107.0)}

```diff
@@ -4,3 +4,3 @@
 
-	if (!*buf || !strcmp(buf, "help")) {
+	if (buf == NULL || !strcmp(buf, "help")) {
 		nsock_printf_nul(sd,
```


## starcoder_lora: same-function pairs 91: correct without bridge 1, with bridge 2; fixes 2, breaks 1, both fail 88


All 469 pairs: P-C 11.30% without vs 15.14% with the bridge | fixes 43, breaks 25, both fail 373


### fixes (0 short; top 2)

- **linux**  ['CWE-200'] commit 817b8b9c53 | 41 lines, 1 changed | without bridge 0.038/0.038 (thr 0.05), with bridge 0.125/0.117 (thr 0.12) | KG slots 32/32, same entities: False | features changed: {'loc': (41.0, 42.0), 'halstead_n2': (55.0, 56.0), 'halstead_N1': (106.0, 109.0), 'halstead_N2': (99.0, 101.0), 'halstead_vol': (1247.93, 1282.79), 'fan_out': (15.0, 16.0)}

```diff
@@ -38,2 +38,3 @@
 err_free:
+	usb_put_dev(udev);
 	kfree(priv);
```

- **tensorflow**  ['CWE-354'] commit e7f497570a | 83 lines, 108 changed | without bridge 0.102/0.067 (thr 0.05), with bridge 0.664/0.050 (thr 0.12) | KG slots 32/32, same entities: False | features changed: {'loc': (83.0, 37.0), 'n_derefs': (7.0, 2.0), 'n_arrays': (7.0, 0.0), 'n_neg_check': (3.0, 0.0), 'halstead_n1': (20.0, 17.0), 'halstead_n2': (133.0, 89.0), 'halstead_N1': (416.0, 132.0), 'halstead_N2': (391.0, 169.0), 'halstead_vol': (5856.71, 2025.1), 'fan_out': (18.0, 11.0)}

```diff
@@ -1,83 +1,37 @@
-void operator()(OpKernelContext* ctx, const Tensor& input,
-                  const Tensor& filter, int row_stride, int col_stride,
-                  int row_dilation, int col_dilation, const Padding& padding,
+void operator()(OpKernelContext* ctx, bool use_cudnn, bool cudnn_use_autotune,
+                  const Tensor& input, const Tensor& filter, int row_dilation,
+                  int col_dilation, int row_stride, int col_stride,
+                  const Padding& padding,
                   const std::vector<int64_t>& explicit_paddings, Tensor* output,
                   TensorFormat data_format) {
-    DCHECK(data_format == FORMAT_NHWC)
-        << "Grouped conv implementation only "
-           "supports NHWC tensor format for now.";
+    if (data_format != FORMAT_NHWC) {
+      ctx->SetStatus(
+          errors::Unimplemented("The Conv2D op currently only supports the "
+                                "NHWC tensor format for integer types. "
+                                "The op was given the format: ",
+                                ToString(data_format)));
+      return;
+    }
+    const int64_t in_depth = GetTensorDim(input, data_format, 'C');
+    OP_REQUIRES(ctx, in_depth == filter.dim_size(2),
+                errors::Unimplemented(
+                    "The Conv2D op currently does not support grouped "
+                    "convolutions for integer types. A grouped convolution was "
+                    "attempted to be run because the input depth of ",
+                    in_depth, " does not match the filter input depth of ",
+                    filter.dim_size(2)));
+    OP_REQUIRES(
+        ctx, filter.NumElements() > 0,
+        errors::InvalidArgument("filter must not have zero elements "
+                                "(i.e. all dimensions must be non-zero)"));
 
-    const int64_t in_depth = input.dim_size(3);
-    const int64_t patch_depth = filter.dim_size(2);
-    const int64_t num_groups = in_depth / patch_depth;
-
```


### breaks (0 short; top 1)

- **tensorflow**  ['CWE-416'] commit e746adbfcf | 95 lines, 1 changed | without bridge 0.052/0.047 (thr 0.05), with bridge 0.056/0.087 (thr 0.12) | KG slots 32/32, same entities: True | features changed: {'loc': (95.0, 94.0), 'halstead_N1': (441.0, 435.0), 'halstead_N2': (487.0, 484.0), 'halstead_vol': (7245.23, 7174.96)}

```diff
@@ -20,3 +20,2 @@
         height <= 0 || height >= (1LL << 27) || total_size >= (1LL << 29)) {
-      png::CommonFreeDecode(&decode);
       OP_REQUIRES(context, false,
```


### both_fail (12 short; top 5)

- **linux**  ['CWE-834'] commit 775c5033a0 | 4 lines, 1 changed | without bridge 0.031/0.031 (thr 0.05), with bridge 0.014/0.014 (thr 0.12) | KG slots 32/32, same entities: False | features changed: {'loc': (4.0, 5.0), 'halstead_n2': (10.0, 11.0), 'halstead_N1': (13.0, 16.0), 'halstead_N2': (12.0, 14.0), 'halstead_vol': (106.2, 129.66), 'fan_out': (3.0, 4.0)}

```diff
@@ -2,2 +2,3 @@
 {
+	remove_inode_hash(inode);
 	set_bit(FUSE_I_BAD, &get_fuse_inode(inode)->state);
```

- **tensorflow**  ['CWE-125'] commit 701cfaca22 | 8 lines, 2 changed | without bridge 0.042/0.044 (thr 0.05), with bridge 0.019/0.019 (thr 0.12) | KG slots 8/10, same entities: False | features changed: {'loc': (8.0, 10.0), 'n_derefs': (9.0, 11.0), 'halstead_n1': (11.0, 12.0), 'halstead_n2': (22.0, 26.0), 'halstead_N1': (55.0, 68.0), 'halstead_N2': (44.0, 54.0), 'halstead_vol': (499.4, 640.25), 'fan_out': (8.0, 10.0)}

```diff
@@ -1,2 +1,4 @@
 Status SparseCountSparseOutputShapeFn(InferenceContext *c) {
+  ShapeHandle unused;
+  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &unused));
   auto rank = c->Dim(c->input(0), 1);
```

- **qemu**  ['CWE-835'] commit 1caff0340f | 14 lines, 2 changed | without bridge 0.034/0.033 (thr 0.05), with bridge 0.026/0.027 (thr 0.12) | KG slots 32/32, same entities: False | features changed: {'n_derefs': (5.0, 3.0), 'halstead_n2': (32.0, 31.0), 'halstead_N1': (57.0, 55.0), 'halstead_N2': (48.0, 46.0), 'halstead_vol': (573.24, 548.05)}

```diff
@@ -7,3 +7,3 @@
     if (s->phy_reg[PHY_CTRL] & MII_CR_LOOPBACK) {
-        nc->info->receive(nc, buf, size);
+        qemu_receive_packet(nc, buf, size);
     } else {
```

- **vim**  ['CWE-476'] commit 8e4b76da1d | 26 lines, 2 changed | without bridge 0.033/0.033 (thr 0.05), with bridge 0.024/0.025 (thr 0.12) | KG slots 4/4, same entities: True | features changed: {'cyclomatic': (5.0, 6.0), 'n_derefs': (1.0, 2.0), 'n_null_check': (2.0, 3.0), 'halstead_n2': (41.0, 42.0), 'halstead_N1': (65.0, 69.0), 'halstead_N2': (70.0, 74.0), 'halstead_vol': (780.48, 830.45)}

```diff
@@ -14,3 +14,3 @@
 	    match = name;
-	else
+	else if (rmp->regprog != NULL)
 	{
```

- **nagioscore**  ['CWE-476'] commit b1a92a3b52 | 33 lines, 2 changed | without bridge 0.035/0.035 (thr 0.05), with bridge 0.014/0.015 (thr 0.12) | KG slots 16/16, same entities: True | features changed: {'n_null_check': (2.0, 3.0), 'halstead_N1': (103.0, 102.0), 'halstead_N2': (106.0, 107.0)}

```diff
@@ -4,3 +4,3 @@
 
-	if (!*buf || !strcmp(buf, "help")) {
+	if (buf == NULL || !strcmp(buf, "help")) {
 		nsock_printf_nul(sd,
```


## starcoder2_lora: same-function pairs 91: correct without bridge 3, with bridge 1; fixes 1, breaks 3, both fail 87


All 469 pairs: P-C 13.86% without vs 16.42% with the bridge | fixes 39, breaks 27, both fail 365


### fixes (0 short; top 1)

- **libjpeg**  ['CWE-703'] commit 4746b57793 | 170 lines, 189 changed | without bridge 0.009/0.002 (thr 0.55), with bridge 0.247/0.129 (thr 0.23) | KG slots 32/32, same entities: False | features changed: {'loc': (170.0, 151.0), 'cyclomatic': (23.0, 14.0), 'max_depth': (7.0, 6.0), 'n_returns': (1.0, 2.0), 'n_arrays': (62.0, 8.0), 'n_format': (0.0, 4.0), 'halstead_n2': (231.0, 214.0), 'halstead_N1': (644.0, 488.0), 'halstead_N2': (738.0, 612.0), 'halstead_vol': (11063.77, 8697.58), 'fan_out': (25.0, 27.0), 'has_recursion': (0.0, 1.0)}

```diff
@@ -1,3 +1,3 @@
-bool SampleInterleavedLSScan::ParseMCU(void)
-{
+bool SingleComponentLSScan::ParseMCU(void)
+{ 
 #if ACCUSOFT_CODE
@@ -5,5 +5,4 @@
   UBYTE preshift        = m_ucLowBit + FractionalColorBitsOf();
-  struct Line *line[4];
-  UBYTE cx;
-
+  struct Line *line     = CurrentLine(0);
+  
   //
@@ -14,2 +13,5 @@
   }
+
+  assert(m_ucCount == 1);
+
   //
@@ -21,10 +23,5 @@
     m_ulRemaining[0] -= lines;
-  assert(lines > 0);
-  assert(m_ucCount < 4);
 
-  //
-  // Fill the line pointers.
-  for(cx = 0;cx < m_ucCount;cx++) {
-    line[cx] = CurrentLine(cx);
-  }
+  if (lines == 0)
+    return false;
 
@@ -33,31 +30,22 @@
     LONG length = m_ulWidth[0];
-    LONG *lp[4];
+    LONG *lp    = line->m_pData;
 
```


### breaks (0 short; top 3)

- **tensorflow**  ['CWE-703'] commit 3150642acb | 231 lines, 5 changed | without bridge 0.654/0.298 (thr 0.55), with bridge 0.607/0.623 (thr 0.23) | KG slots 32/32, same entities: True | features changed: {'loc': (231.0, 236.0), 'n_derefs': (15.0, 17.0), 'halstead_n2': (317.0, 318.0), 'halstead_N1': (850.0, 871.0), 'halstead_N2': (1008.0, 1030.0), 'halstead_vol': (15679.28, 16050.04)}

```diff
@@ -7,2 +7,7 @@
     OP_REQUIRES_OK(context, context->input("row_remapping", &row_remapping_t));
+    OP_REQUIRES(
+        context, row_remapping_t->dims() == 1,
+        errors::InvalidArgument("The `row_remapping` tensor must be 1-D, got "
+                                "a tensor of shape ",
+                                row_remapping_t->shape().DebugString()));
     const auto row_remapping = row_remapping_t->vec<int64_t>();
```

- **php-src**  ['CWE-476'] commit 698a691724 | 136 lines, 8 changed | without bridge 0.619/0.495 (thr 0.55), with bridge 0.139/0.192 (thr 0.23) | KG slots 32/32, same entities: False | features changed: {'loc': (136.0, 140.0), 'cyclomatic': (18.0, 19.0), 'n_derefs': (52.0, 53.0), 'halstead_n2': (126.0, 127.0), 'halstead_N1': (555.0, 565.0), 'halstead_N2': (396.0, 402.0), 'halstead_vol': (6828.09, 6952.56), 'fan_out': (33.0, 34.0)}

```diff
@@ -39,4 +39,8 @@
 			STR_FREE(Z_STRVAL_P(ent1->data));
-			Z_STRVAL_P(ent1->data) = new_str;
-			Z_STRLEN_P(ent1->data) = new_len;
+			if (new_str) {
+				Z_STRVAL_P(ent1->data) = new_str;
+				Z_STRLEN_P(ent1->data) = new_len;
+			} else {
+				ZVAL_EMPTY_STRING(ent1->data);
+			}
 		}
```

- **tensorflow**  ['CWE-125'] commit 002408c369 | 141 lines, 232 changed | without bridge 0.894/0.393 (thr 0.55), with bridge 0.683/0.874 (thr 0.23) | KG slots 32/32, same entities: False | features changed: {'loc': (141.0, 119.0), 'cyclomatic': (8.0, 9.0), 'max_depth': (7.0, 6.0), 'n_arrays': (0.0, 38.0), 'n_ptr_arith': (2.0, 0.0), 'n_null_check': (4.0, 0.0), 'halstead_n1': (24.0, 25.0), 'halstead_n2': (200.0, 126.0), 'halstead_N1': (560.0, 547.0), 'halstead_N2': (637.0, 448.0), 'halstead_vol': (9345.4, 7202.21), 'fan_out': (29.0, 30.0)}

```diff
@@ -1,125 +1,112 @@
 void Compute(OpKernelContext* context) override {
-    // Here's the basic idea:
-    // Batch and depth dimension are independent from row and col dimension. And
-    // because FractionalAvgPool currently only support pooling along row and
-    // col, we can basically think of this 4D tensor backpropagation as
-    // operation of a series of 2D planes.
-    //
-    // For each element of a 'slice' (2D plane) of output_backprop, we need to
-    // figure out its contributors when doing FractionalAvgPool operation. This
-    // can be done based on row_pooling_sequence, col_pooling_seq and
-    // overlapping.
-    // Once we figure out the original contributors, we just need to evenly
-    // divide the value of this element among these contributors.
-    //
-    // Internally, we divide the out_backprop tensor and store it in a temporary
-    // tensor of double type. And cast it to the corresponding type.
     typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
         ConstEigenMatrixMap;
-    typedef Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
-        EigenDoubleMatrixMap;
-
-    // Grab the inputs.
-    const Tensor& orig_input_tensor_shape = context->input(0);
-    OP_REQUIRES(context,
-                orig_input_tensor_shape.dims() == 1 &&
-                    orig_input_tensor_shape.NumElements() == 4,
-                errors::InvalidArgument("original input tensor shape must be"
-                                        "1-dimensional and 4 elements"));
-    const Tensor& out_backprop = context->input(1);
-    const Tensor& row_seq_tensor = context->input(2);
-    const Tensor& col_seq_tensor = context->input(3);
-
-    const int64_t out_batch = out_backprop.dim_size(0);
-    const int64_t out_rows = out_backprop.dim_size(1);
-    const int64_t out_cols = out_backprop.dim_size(2);
-    const int64_t out_depth = out_backprop.dim_size(3);
-
```


### both_fail (12 short; top 5)

- **linux**  ['CWE-834'] commit 775c5033a0 | 4 lines, 1 changed | without bridge 0.000/0.000 (thr 0.55), with bridge 0.012/0.009 (thr 0.23) | KG slots 32/32, same entities: False | features changed: {'loc': (4.0, 5.0), 'halstead_n2': (10.0, 11.0), 'halstead_N1': (13.0, 16.0), 'halstead_N2': (12.0, 14.0), 'halstead_vol': (106.2, 129.66), 'fan_out': (3.0, 4.0)}

```diff
@@ -2,2 +2,3 @@
 {
+	remove_inode_hash(inode);
 	set_bit(FUSE_I_BAD, &get_fuse_inode(inode)->state);
```

- **tensorflow**  ['CWE-125'] commit 701cfaca22 | 8 lines, 2 changed | without bridge 0.000/0.000 (thr 0.55), with bridge 0.058/0.066 (thr 0.23) | KG slots 8/10, same entities: False | features changed: {'loc': (8.0, 10.0), 'n_derefs': (9.0, 11.0), 'halstead_n1': (11.0, 12.0), 'halstead_n2': (22.0, 26.0), 'halstead_N1': (55.0, 68.0), 'halstead_N2': (44.0, 54.0), 'halstead_vol': (499.4, 640.25), 'fan_out': (8.0, 10.0)}

```diff
@@ -1,2 +1,4 @@
 Status SparseCountSparseOutputShapeFn(InferenceContext *c) {
+  ShapeHandle unused;
+  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &unused));
   auto rank = c->Dim(c->input(0), 1);
```

- **qemu**  ['CWE-835'] commit 1caff0340f | 14 lines, 2 changed | without bridge 0.000/0.000 (thr 0.55), with bridge 0.022/0.022 (thr 0.23) | KG slots 32/32, same entities: False | features changed: {'n_derefs': (5.0, 3.0), 'halstead_n2': (32.0, 31.0), 'halstead_N1': (57.0, 55.0), 'halstead_N2': (48.0, 46.0), 'halstead_vol': (573.24, 548.05)}

```diff
@@ -7,3 +7,3 @@
     if (s->phy_reg[PHY_CTRL] & MII_CR_LOOPBACK) {
-        nc->info->receive(nc, buf, size);
+        qemu_receive_packet(nc, buf, size);
     } else {
```

- **vim**  ['CWE-476'] commit 8e4b76da1d | 26 lines, 2 changed | without bridge 0.200/0.086 (thr 0.55), with bridge 0.041/0.038 (thr 0.23) | KG slots 4/4, same entities: True | features changed: {'cyclomatic': (5.0, 6.0), 'n_derefs': (1.0, 2.0), 'n_null_check': (2.0, 3.0), 'halstead_n2': (41.0, 42.0), 'halstead_N1': (65.0, 69.0), 'halstead_N2': (70.0, 74.0), 'halstead_vol': (780.48, 830.45)}

```diff
@@ -14,3 +14,3 @@
 	    match = name;
-	else
+	else if (rmp->regprog != NULL)
 	{
```

- **nagioscore**  ['CWE-476'] commit b1a92a3b52 | 33 lines, 2 changed | without bridge 0.000/0.000 (thr 0.55), with bridge 0.024/0.022 (thr 0.23) | KG slots 16/16, same entities: True | features changed: {'n_null_check': (2.0, 3.0), 'halstead_N1': (103.0, 102.0), 'halstead_N2': (106.0, 107.0)}

```diff
@@ -4,3 +4,3 @@
 
-	if (!*buf || !strcmp(buf, "help")) {
+	if (buf == NULL || !strcmp(buf, "help")) {
 		nsock_printf_nul(sd,
```


## qwen_lora: same-function pairs 91: correct without bridge 2, with bridge 1; fixes 0, breaks 1, both fail 89


All 469 pairs: P-C 15.14% without vs 15.14% with the bridge | fixes 22, breaks 22, both fail 376


### fixes (0 short; top 0)


### breaks (0 short; top 1)

- **tensorflow**  ['CWE-416'] commit e746adbfcf | 95 lines, 1 changed | without bridge 0.406/0.125 (thr 0.3), with bridge 0.240/0.252 (thr 0.23) | KG slots 32/32, same entities: True | features changed: {'loc': (95.0, 94.0), 'halstead_N1': (441.0, 435.0), 'halstead_N2': (487.0, 484.0), 'halstead_vol': (7245.23, 7174.96)}

```diff
@@ -20,3 +20,2 @@
         height <= 0 || height >= (1LL << 27) || total_size >= (1LL << 29)) {
-      png::CommonFreeDecode(&decode);
       OP_REQUIRES(context, false,
```


### both_fail (12 short; top 5)

- **linux**  ['CWE-834'] commit 775c5033a0 | 4 lines, 1 changed | without bridge 0.000/0.000 (thr 0.3), with bridge 0.032/0.033 (thr 0.23) | KG slots 32/32, same entities: False | features changed: {'loc': (4.0, 5.0), 'halstead_n2': (10.0, 11.0), 'halstead_N1': (13.0, 16.0), 'halstead_N2': (12.0, 14.0), 'halstead_vol': (106.2, 129.66), 'fan_out': (3.0, 4.0)}

```diff
@@ -2,2 +2,3 @@
 {
+	remove_inode_hash(inode);
 	set_bit(FUSE_I_BAD, &get_fuse_inode(inode)->state);
```

- **tensorflow**  ['CWE-125'] commit 701cfaca22 | 8 lines, 2 changed | without bridge 0.006/0.011 (thr 0.3), with bridge 0.062/0.068 (thr 0.23) | KG slots 8/10, same entities: False | features changed: {'loc': (8.0, 10.0), 'n_derefs': (9.0, 11.0), 'halstead_n1': (11.0, 12.0), 'halstead_n2': (22.0, 26.0), 'halstead_N1': (55.0, 68.0), 'halstead_N2': (44.0, 54.0), 'halstead_vol': (499.4, 640.25), 'fan_out': (8.0, 10.0)}

```diff
@@ -1,2 +1,4 @@
 Status SparseCountSparseOutputShapeFn(InferenceContext *c) {
+  ShapeHandle unused;
+  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &unused));
   auto rank = c->Dim(c->input(0), 1);
```

- **qemu**  ['CWE-835'] commit 1caff0340f | 14 lines, 2 changed | without bridge 0.009/0.043 (thr 0.3), with bridge 0.050/0.051 (thr 0.23) | KG slots 32/32, same entities: False | features changed: {'n_derefs': (5.0, 3.0), 'halstead_n2': (32.0, 31.0), 'halstead_N1': (57.0, 55.0), 'halstead_N2': (48.0, 46.0), 'halstead_vol': (573.24, 548.05)}

```diff
@@ -7,3 +7,3 @@
     if (s->phy_reg[PHY_CTRL] & MII_CR_LOOPBACK) {
-        nc->info->receive(nc, buf, size);
+        qemu_receive_packet(nc, buf, size);
     } else {
```

- **vim**  ['CWE-476'] commit 8e4b76da1d | 26 lines, 2 changed | without bridge 0.168/0.193 (thr 0.3), with bridge 0.082/0.088 (thr 0.23) | KG slots 4/4, same entities: True | features changed: {'cyclomatic': (5.0, 6.0), 'n_derefs': (1.0, 2.0), 'n_null_check': (2.0, 3.0), 'halstead_n2': (41.0, 42.0), 'halstead_N1': (65.0, 69.0), 'halstead_N2': (70.0, 74.0), 'halstead_vol': (780.48, 830.45)}

```diff
@@ -14,3 +14,3 @@
 	    match = name;
-	else
+	else if (rmp->regprog != NULL)
 	{
```

- **nagioscore**  ['CWE-476'] commit b1a92a3b52 | 33 lines, 2 changed | without bridge 0.000/0.000 (thr 0.3), with bridge 0.043/0.042 (thr 0.23) | KG slots 16/16, same entities: True | features changed: {'n_null_check': (2.0, 3.0), 'halstead_N1': (103.0, 102.0), 'halstead_N2': (106.0, 107.0)}

```diff
@@ -4,3 +4,3 @@
 
-	if (!*buf || !strcmp(buf, "help")) {
+	if (buf == NULL || !strcmp(buf, "help")) {
 		nsock_printf_nul(sd,
```


## deepseek_lora: same-function pairs 91: correct without bridge 4, with bridge 11; fixes 9, breaks 2, both fail 78


All 469 pairs: P-C 17.91% without vs 16.42% with the bridge | fixes 38, breaks 45, both fail 347


### fixes (1 short; top 1)

- **tensorflow**  ['CWE-125'] commit 701cfaca22 | 8 lines, 2 changed | without bridge 0.000/0.003 (thr 0.02), with bridge 0.772/0.000 (thr 0.03) | KG slots 8/10, same entities: False | features changed: {'loc': (8.0, 10.0), 'n_derefs': (9.0, 11.0), 'halstead_n1': (11.0, 12.0), 'halstead_n2': (22.0, 26.0), 'halstead_N1': (55.0, 68.0), 'halstead_N2': (44.0, 54.0), 'halstead_vol': (499.4, 640.25), 'fan_out': (8.0, 10.0)}

```diff
@@ -1,2 +1,4 @@
 Status SparseCountSparseOutputShapeFn(InferenceContext *c) {
+  ShapeHandle unused;
+  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &unused));
   auto rank = c->Dim(c->input(0), 1);
```


### breaks (0 short; top 2)

- **net**  ['CWE-125'] commit 7892032cfe | 77 lines, 40 changed | without bridge 0.029/0.001 (thr 0.02), with bridge 0.003/0.000 (thr 0.03) | KG slots 32/32, same entities: False | features changed: {'loc': (77.0, 79.0), 'n_returns': (3.0, 4.0), 'n_derefs': (20.0, 22.0), 'n_arrays': (3.0, 1.0), 'halstead_n1': (25.0, 24.0), 'halstead_n2': (101.0, 94.0), 'halstead_N1': (250.0, 241.0), 'halstead_N2': (247.0, 246.0), 'halstead_vol': (3467.71, 3351.85), 'has_recursion': (1.0, 0.0)}

```diff
@@ -1,31 +1,33 @@
 static void ip6gre_err(struct sk_buff *skb, struct inet6_skb_parm *opt,
-		u8 type, u8 code, int offset, __be32 info)
+		       u8 type, u8 code, int offset, __be32 info)
 {
-	const struct ipv6hdr *ipv6h = (const struct ipv6hdr *)skb->data;
-	__be16 *p = (__be16 *)(skb->data + offset);
-	int grehlen = offset + 4;
+	const struct gre_base_hdr *greh;
+	const struct ipv6hdr *ipv6h;
+	int grehlen = sizeof(*greh);
 	struct ip6_tnl *t;
+	int key_off = 0;
 	__be16 flags;
+	__be32 key;
 
-	flags = p[0];
-	if (flags&(GRE_CSUM|GRE_KEY|GRE_SEQ|GRE_ROUTING|GRE_VERSION)) {
-		if (flags&(GRE_VERSION|GRE_ROUTING))
-			return;
-		if (flags&GRE_KEY) {
-			grehlen += 4;
-			if (flags&GRE_CSUM)
-				grehlen += 4;
-		}
+	if (!pskb_may_pull(skb, offset + grehlen))
+		return;
+	greh = (const struct gre_base_hdr *)(skb->data + offset);
+	flags = greh->flags;
+	if (flags & (GRE_VERSION | GRE_ROUTING))
+		return;
+	if (flags & GRE_CSUM)
+		grehlen += 4;
+	if (flags & GRE_KEY) {
+		key_off = grehlen + offset;
+		grehlen += 4;
 	}
 
```

- **pjproject**  ['CWE-787'] commit 450baca94f | 65 lines, 65 changed | without bridge 0.249/0.000 (thr 0.02), with bridge 0.001/0.000 (thr 0.03) | KG slots 32/5, same entities: False | features changed: {'loc': (65.0, 14.0), 'cyclomatic': (10.0, 2.0), 'max_depth': (3.0, 2.0), 'n_returns': (4.0, 2.0), 'n_derefs': (13.0, 3.0), 'n_size_t': (1.0, 0.0), 'halstead_n1': (23.0, 14.0), 'halstead_n2': (64.0, 19.0), 'halstead_N1': (197.0, 37.0), 'halstead_N2': (163.0, 31.0), 'halstead_vol': (2319.46, 343.02), 'fan_out': (6.0, 4.0), 'n_signed_cmp': (1.0, 0.0)}

```diff
@@ -1,6 +1,4 @@
-PJ_DEF(pj_status_t) pjstun_parse_msg( void *buf, pj_size_t buf_len, 
-				      pjstun_msg *msg)
+PJ_DEF(void*) pjstun_msg_find_attr( pjstun_msg *msg, pjstun_attr_type t)
 {
-    pj_uint16_t msg_type, msg_len;
-    char *p_attr;
+    int i;
 
@@ -8,58 +6,9 @@
 
-    msg->hdr = (pjstun_msg_hdr*)buf;
-    msg_type = pj_ntohs(msg->hdr->type);
-
-    switch (msg_type) {
-    case PJSTUN_BINDING_REQUEST:
-    case PJSTUN_BINDING_RESPONSE:
-    case PJSTUN_BINDING_ERROR_RESPONSE:
-    case PJSTUN_SHARED_SECRET_REQUEST:
-    case PJSTUN_SHARED_SECRET_RESPONSE:
-    case PJSTUN_SHARED_SECRET_ERROR_RESPONSE:
-	break;
-    default:
-	PJ_LOG(4,(THIS_FILE, "Error: unknown msg type %d", msg_type));
-	return PJLIB_UTIL_ESTUNINMSGTYPE;
+    for (i=0; i<msg->attr_count; ++i) {
+	pjstun_attr_hdr *attr = msg->attr[i];
+	if (pj_ntohs(attr->type) == t)
+	    return attr;
     }
 
-    msg_len = pj_ntohs(msg->hdr->length);
-    if (msg_len != buf_len - sizeof(pjstun_msg_hdr)) {
-	PJ_LOG(4,(THIS_FILE, "Error: invalid msg_len %d (expecting %d)", 
-			     msg_len, buf_len - sizeof(pjstun_msg_hdr)));
-	return PJLIB_UTIL_ESTUNINMSGLEN;
-    }
-
```


### both_fail (11 short; top 5)

- **linux**  ['CWE-834'] commit 775c5033a0 | 4 lines, 1 changed | without bridge 0.000/0.000 (thr 0.02), with bridge 0.000/0.000 (thr 0.03) | KG slots 32/32, same entities: False | features changed: {'loc': (4.0, 5.0), 'halstead_n2': (10.0, 11.0), 'halstead_N1': (13.0, 16.0), 'halstead_N2': (12.0, 14.0), 'halstead_vol': (106.2, 129.66), 'fan_out': (3.0, 4.0)}

```diff
@@ -2,2 +2,3 @@
 {
+	remove_inode_hash(inode);
 	set_bit(FUSE_I_BAD, &get_fuse_inode(inode)->state);
```

- **qemu**  ['CWE-835'] commit 1caff0340f | 14 lines, 2 changed | without bridge 0.000/0.000 (thr 0.02), with bridge 0.000/0.000 (thr 0.03) | KG slots 32/32, same entities: False | features changed: {'n_derefs': (5.0, 3.0), 'halstead_n2': (32.0, 31.0), 'halstead_N1': (57.0, 55.0), 'halstead_N2': (48.0, 46.0), 'halstead_vol': (573.24, 548.05)}

```diff
@@ -7,3 +7,3 @@
     if (s->phy_reg[PHY_CTRL] & MII_CR_LOOPBACK) {
-        nc->info->receive(nc, buf, size);
+        qemu_receive_packet(nc, buf, size);
     } else {
```

- **vim**  ['CWE-476'] commit 8e4b76da1d | 26 lines, 2 changed | without bridge 0.000/0.000 (thr 0.02), with bridge 0.000/0.000 (thr 0.03) | KG slots 4/4, same entities: True | features changed: {'cyclomatic': (5.0, 6.0), 'n_derefs': (1.0, 2.0), 'n_null_check': (2.0, 3.0), 'halstead_n2': (41.0, 42.0), 'halstead_N1': (65.0, 69.0), 'halstead_N2': (70.0, 74.0), 'halstead_vol': (780.48, 830.45)}

```diff
@@ -14,3 +14,3 @@
 	    match = name;
-	else
+	else if (rmp->regprog != NULL)
 	{
```

- **nagioscore**  ['CWE-476'] commit b1a92a3b52 | 33 lines, 2 changed | without bridge 0.000/0.000 (thr 0.02), with bridge 0.000/0.000 (thr 0.03) | KG slots 16/16, same entities: True | features changed: {'n_null_check': (2.0, 3.0), 'halstead_N1': (103.0, 102.0), 'halstead_N2': (106.0, 107.0)}

```diff
@@ -4,3 +4,3 @@
 
-	if (!*buf || !strcmp(buf, "help")) {
+	if (buf == NULL || !strcmp(buf, "help")) {
 		nsock_printf_nul(sd,
```

- **libvncserver**  ['CWE-787'] commit 7b1ef0ffc4 | 36 lines, 2 changed | without bridge 0.003/0.001 (thr 0.02), with bridge 0.000/0.000 (thr 0.03) | KG slots 32/32, same entities: True | features changed: {}

```diff
@@ -18,3 +18,3 @@
 
-    if (hdr.nSubrects * (4 + (BPP / 8)) > RFB_BUFFER_SIZE || !ReadFromRFBServer(client, client->buffer, hdr.nSubrects * (4 + (BPP / 8))))
+    if (hdr.nSubrects > RFB_BUFFER_SIZE / (4 + (BPP / 8)) || !ReadFromRFBServer(client, client->buffer, hdr.nSubrects * (4 + (BPP / 8))))
 	return FALSE;
```
