
use blockless_sdk::{BlocklessHttp, HttpOptions};
use candle_core::{Device, Tensor, Result};
use candle_onnx::onnx;
use candle_onnx::onnx::{AttributeProto, GraphProto, ModelProto, NodeProto, ValueInfoProto};
use std::collections::HashMap;
use blockless_sdk::create_model_proto_with_graph;
// use protobuf::Message;
use prost::Message;
// use onnx::onnx::ModelProto;
// use std::fs::File;
// use std::io::{BufReader, Read};
// use anyhow::Result;
// use protobuf::Message;

// const PROMPT: &str = "The corsac fox (Vulpes corsac), also known simply as a corsac, is a medium-sized fox found in";
// /// Max tokens to generate
// const GEN_TOKENS: i32 = 90;
// /// Top_K -> Sample from the k most likely next tokens at each step. Lower k focuses on higher probability tokens.
// const TOP_K: usize = 5;

/// GPT-2 Text Generation
///22
/// This Rust program demonstrates text generation using the GPT-2 language model with `ort`.
/// The program initializes the model, tokenizes a prompt, and generates a sequence of tokens.
/// It utilizes top-k sampling for diverse and contextually relevant text generation.


///// ----------------- ONNX.rs -----------------

// pub struct OnnxModelInfo {
//     pub input_names: Vec<String>,
//     pub output_names: Vec<String>,
//     pub model_name: String,
//     pub model_ir_version: i64,
// }

// pub fn load_model_and_get_info<P: AsRef<std::path::Path>>(model_path: P) -> Result<OnnxModelInfo> {
//     // Open the ONNX model file
//     let file = File::open(model_path)?;
//     let mut reader = BufReader::new(file);

//     // Read the file into a buffer
//     let mut buffer = Vec::new();
//     reader.read_to_end(&mut buffer)?;

//     // Create a new ModelProto instance
//     let mut model_proto = ModelProto::new();

//     // Parse the buffer into the ModelProto instance
//     let mut coded_input_stream = protobuf::CodedInputStream::from_bytes(&buffer);
//     protobuf::Message::merge_from(&mut model_proto, &mut coded_input_stream)?;
//     model_proto.merge_from(&mut coded_input_stream)?;

//     // Extract model information
//     let model_name = model_proto.get_graph().get_name().to_string();
//     let model_ir_version = model_proto.get_ir_version();

//     // Extract input names
//     let input_names = model_proto
//         .get_graph()
//         .get_input()
//         .iter()
//         .map(|input| input.get_name().to_string())
//         .collect::<Vec<String>>();

//     // Extract output names
//     let output_names = model_proto
//         .get_graph()
//         .get_output()
//         .iter()
//         .map(|output| output.get_name().to_string())
//         .collect::<Vec<String>>();

//     Ok(OnnxModelInfo {
//         input_names,
//         output_names,
//         model_name,
//         model_ir_version,
//     })
// }


const INPUT_X: &str = "x";
const INPUT_Y: &str = "y";
const INPUT_A: &str = "a";
const OUTPUT_Z: &str = "z";

fn test_add_operation() -> Result<f64> {
    let manual_graph = create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Add".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string(), INPUT_Y.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), Tensor::new(&[2.], &Device::Cpu)?);
    inputs.insert(INPUT_Y.to_string(), Tensor::new(&[2.], &Device::Cpu)?);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");
    let first = z.to_vec1::<f64>()?[0];
    assert_eq!(first, 4.0f64);
    Ok(first)
}

fn main() -> Result<()> {

    // let device = Device::Cpu;

    // let a = Tensor::randn(0f32, 1., (2, 3), &device)?;
    // let b = Tensor::randn(0f32, 1., (3, 4), &device)?;

    // let c = a.matmul(&b)?;
    // println!("{c}");

	// let res = test_add_operation().unwrap();
	// println!("{res}");


    // let socket = create_tcp_bind_socket("127.0.0.1:4000");


    //// ----------------- ORT -----------------
    
	// // Load our model
	// let session = Session::builder().unwrap()
	// 	.with_optimization_level(GraphOptimizationLevel::Level1).unwrap()
	// 	.with_intra_threads(1).unwrap()
	// 	.commit_from_url("https://parcel.pyke.io/v2/cdn/assetdelivery/ortrsv2/ex_models/gpt2.onnx").unwrap();

	// // Load the tokenizer and encode the prompt into a sequence of tokens.
	// let tokenizer = Tokenizer::from_file(Path::new(env!("CARGO_MANIFEST_DIR")).join("data").join("tokenizer.json")).unwrap();
	// let tokens = tokenizer.encode(PROMPT, false).unwrap();
	// let tokens = tokens.get_ids().iter().map(|i| *i as i64).collect::<Vec<_>>();

	// let mut tokens = Array1::from_iter(tokens.iter().cloned());

	// print!("{PROMPT}");
	// stdout.flush().unwrap();

	// for _ in 0..GEN_TOKENS {
	// 	let array = tokens.view().insert_axis(Axis(0)).insert_axis(Axis(1));
	// 	let outputs = session.run(inputs![array].unwrap()).unwrap();
	// 	let generated_tokens: ArrayViewD<f32> = outputs["output1"].try_extract_tensor().unwrap();

	// 	// Collect and sort logits
	// 	let probabilities = &mut generated_tokens
	// 		.slice(s![0, 0, -1, ..])
	// 		.insert_axis(Axis(0))
	// 		.to_owned()
	// 		.iter()
	// 		.cloned()
	// 		.enumerate()
	// 		.collect::<Vec<_>>();
	// 	probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Less));

	// 	// Sample using top-k sampling
	// 	let token = probabilities[rng.gen_range(0..=TOP_K)].0;
	// 	tokens = concatenate![Axis(0), tokens, array![token.try_into().unwrap()]];

	// 	let token_str = tokenizer.decode(&[token as _], true).unwrap();
	// 	print!("{}", token_str);
	// 	stdout.flush().unwrap();
	// }

	// println!();


    //// ----------------- HTTP -----------------

    
    // let opts = HttpOptions::new("GET", 30, 10);
    // let http = BlocklessHttp::open("https://parcel.pyke.io/v2/cdn/assetdelivery/ortrsv2/ex_models/gpt2.onnx", &opts);
    
    // println!("http: {:?}", http.as_ref().err());
    // let http = http.unwrap();
    // let body = http.get_all_body().unwrap();

	// println!("Downloaded body size: {}", body.len());
	// println!("First 100 bytes of body: {:?}", &body[..100]);

	let model_bytes = std::fs::read("gpt2.onnx").unwrap();

	// load file into byte buffer at compile time
	// let model_bytes: &[u8] = include_bytes!("../gpt2.onnx");

	let graph: ModelProto = prost::Message::decode(model_bytes.as_slice()).unwrap();

    println!("model input: {:?}", graph.graph.clone().unwrap().input);
    println!("model output: {:?}", graph.graph.clone().unwrap().output);
    println!("layer count: {:?}", graph.graph.clone().unwrap().node.len());
    
    // let body = String::from_utf8(body).unwrap();
    // let bodies = match json::parse(&body).unwrap() {
    //     json::JsonValue::Object(o) => o,
    //     _ => panic!("must be object"),
    // };
    // let headers = match bodies.get("headers") {
    //     Some(json::JsonValue::Object(headers)) => headers,
    //     _ => panic!("must be array"),
    // };
    // headers.iter().for_each(|s| {
    //     println!("{} = {}", s.0, s.1);
    // });
	Ok(())
}
