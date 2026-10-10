// Frozen pre-prepared-plan encode oracle (merge base 7f39ac4), renamed for tests.
use super::*;

impl Model {
    pub(super) fn encode_original(&self, s: &Workspace, original_rope: bool) -> Result<()> {
        let (b, l) = (s.b, s.l);
        let z = self.w("zeros.1024").ptr();
        let attention = |label: &str| {
            if l == 512 && (b == 1 || b == 4) {
                format!("attn_{label}_b{b}_l512")
            } else {
                format!("attn_{label}")
            }
        };
        // All pointers refer to checked fixed-shape, resident allocations in this worker.
        let call = |name: &str, args: &[Ptr]| unsafe { self.cuda.launch(name, args, b, l) };
        call(
            "embed",
            &[
                s.ids.ptr(),
                self.w("encoder.embeddings.tok_embeddings.weight").ptr(),
                self.w("encoder.embeddings.norm.weight").ptr(),
                s.x.ptr(),
                s.y.ptr(),
            ],
        )?;
        self.dump("embedding", &s.x, false)?;
        for i in 0..28 {
            let p = format!("encoder.layers.{i}");
            let w = |n: &str| self.w(&format!("{p}.{n}")).ptr();
            call(
                "qkv",
                &[
                    s.y.ptr(),
                    w("attn.Wqkv.weight"),
                    self.w("zeros.3072").ptr(),
                    s.qkv.ptr(),
                ],
            )?;
            let kind = if i % 3 == 0 { "full" } else { "local" };
            call(
                if original_rope {
                    "rope_original"
                } else {
                    "rope"
                },
                &[
                    s.qkv.ptr(),
                    self.w(&format!("rope_{kind}_cos")).ptr(),
                    self.w(&format!("rope_{kind}_sin")).ptr(),
                ],
            )?;
            call(
                &attention(if i % 3 == 0 { "full" } else { "local" }),
                &[s.qkv.ptr(), s.lens.ptr(), s.o.ptr()],
            )?;
            call("out", &[s.o.ptr(), w("attn.Wo.weight"), z, s.y.ptr()])?;
            call(
                "addln",
                &[s.x.ptr(), s.y.ptr(), w("mlp_norm.weight"), z, s.y.ptr()],
            )?;
            call("geglu", &[s.y.ptr(), w("mlp.Wi.weight"), s.g.ptr()])?;
            call("down", &[s.g.ptr(), w("mlp.Wo.weight"), z, s.y.ptr()])?;
            let next = if i < 27 {
                self.w(&format!("encoder.layers.{}.attn_norm.weight", i + 1))
            } else {
                self.w("encoder.final_norm.weight")
            };
            call("addln", &[s.x.ptr(), s.y.ptr(), next.ptr(), z, s.y.ptr()])?;
            if [0, 1, 2, 27].contains(&i) {
                self.dump(&format!("encoder{i}_residual"), &s.x, false)?;
                self.dump(&format!("encoder{i}_normalized"), &s.y, true)?;
            }
        }
        call(
            "type",
            &[
                s.y.ptr(),
                self.w("type_emb.weight").ptr(),
                s.types.ptr(),
                s.x.ptr(),
            ],
        )?;
        for i in 0..2 {
            let p = format!("head.layers.{i}");
            let w = |n: &str| self.w(&format!("{p}.{n}")).ptr();
            call(
                "ln_bias",
                &[
                    s.x.ptr(),
                    s.y.ptr(),
                    w("norm1.weight"),
                    w("norm1.bias"),
                    s.y.ptr(),
                ],
            )?;
            call(
                "head_in",
                &[
                    s.y.ptr(),
                    w("self_attn.in_proj_weight"),
                    w("self_attn.in_proj_bias"),
                    s.qkv.ptr(),
                ],
            )?;
            call(&attention("full"), &[s.qkv.ptr(), s.lens.ptr(), s.o.ptr()])?;
            call(
                "head_out",
                &[
                    s.o.ptr(),
                    w("self_attn.out_proj.weight"),
                    w("self_attn.out_proj.bias"),
                    s.y.ptr(),
                ],
            )?;
            call(
                "addln_bias",
                &[
                    s.x.ptr(),
                    s.y.ptr(),
                    w("norm2.weight"),
                    w("norm2.bias"),
                    s.y.ptr(),
                ],
            )?;
            call(
                "ffn1",
                &[
                    s.y.ptr(),
                    w("linear1.weight"),
                    w("linear1.bias"),
                    s.ff.ptr(),
                ],
            )?;
            call(
                "ffn2",
                &[
                    s.ff.ptr(),
                    w("linear2.weight"),
                    w("linear2.bias"),
                    s.y.ptr(),
                ],
            )?;
            call("residual", &[s.x.ptr(), s.y.ptr()])?;
        }
        Ok(())
    }
}
