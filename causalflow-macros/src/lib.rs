use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn};

#[proc_macro_attribute]
pub fn causal_tool(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    let name = &input.sig.ident;
    let metadata_name = quote::format_ident!("metadata_for_{}", name);

    let expanded = quote! {
        #input

        pub fn #metadata_name() -> &'static str {
            // Placeholder for JSON metadata generation
            "{ \"name\": \"#name\", \"description\": \"Causal tool defined in Rust\" }"
        }
    };

    TokenStream::from(expanded)
}
