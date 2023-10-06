use std::fs::File;
use std::io::Result;
use std::mem::size_of;
use std::path::Path;

use bytemuck::{Pod, Zeroable};
use ggml_sys_bleedingedge::{ggml_cgraph, ggml_context, ggml_scratch, ggml_tensor, ggml_type};
use memmap2::Mmap;

#[derive(Debug)]
pub struct ModelContext {
    pub dtype: ggml_type,
    pub ctx_w: *mut ggml_context,
    pub ctx_kv: *mut ggml_context,
    pub ctx_b: Option<*mut ggml_context>,
    pub(crate) gf: Option<*mut ggml_cgraph>,
    pub(crate) scratch: ggml_scratch,
    pub(crate) compute_buffer: Vec<char>,
    pub(crate) scratch_buffer: Vec<char>,
    pub(crate) work_buffer: Vec<char>,
}

struct UnInitChar {
    m: char,
}

pub struct ModelLoader {
    file: MappedFile,
    ptr: usize,
}

impl ModelLoader {
    pub fn new(path: &str) -> Result<Self> {
        let f = MappedFile::new(path)?;
        let loader = ModelLoader { file: f, ptr: 0 };
        Ok(loader)
    }

    pub fn read_string(&mut self, offset: usize) -> Option<String> {
        let result = self.read(offset);
        result.map(|data| String::from_utf8_lossy(data).to_string())
    }

    pub fn read_int(&mut self) -> Option<i32> {
        self.read_obj::<i32>()
    }

    pub fn read_obj<T>(&mut self) -> Option<T>
    where
        T: Pod + Zeroable + Copy + Clone,
    {
        let size = size_of::<T>();
        self.read(size)
            .map(|data| bytemuck::cast_slice::<u8, T>(data)[0])
    }

    pub fn read_tensor(&mut self, name: &str, tensor: *mut ggml_tensor) {
        todo!()
    }

    fn read(&mut self, offset: usize) -> Option<&[u8]> {
        let current = self.ptr;
        let p_size = current + offset;
        if p_size <= self.file.data.len() {
            let t = &self.file.data[current..p_size];
            self.ptr = p_size;
            Some(t)
        } else {
            None
        }
    }
}

struct MappedFile {
    data: Mmap,
}

impl MappedFile {
    fn new(path: &str) -> Result<Self> {
        let file = File::open(Path::new(path))?;
        let m = unsafe { Mmap::map(&file)? };
        Ok(MappedFile { data: m })
    }
}

struct Pipeline {}

#[derive(Debug)]
struct GenerationConfig {
    max_length: i64,
    do_sample: bool,
    top_k: i32,
    top_p: f32,
    temperature: f32,
    repetition_penalty: f32,
    num_threads: i32,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        let cpus = num_cpus::get_physical();
        GenerationConfig {
            max_length: 2048,
            do_sample: true,
            top_k: 0,
            top_p: 0.7,
            temperature: 0.95,
            repetition_penalty: 1.,
            num_threads: cpus as i32,
        }
    }
}
