use std::{fs::File, os::windows::io::AsRawHandle};
use windows_sys::Win32::{
    Foundation::CloseHandle,
    System::Memory::{CreateFileMappingW, FILE_MAP_READ, MapViewOfFile, PAGE_READONLY},
};

pub fn mmap_file(f: &File) -> std::io::Result<&[u8]> {
    unsafe {
        let map = CreateFileMappingW(
            f.as_raw_handle(),
            std::ptr::null_mut(),
            PAGE_READONLY,
            0,
            0,
            std::ptr::null(),
        );

        if map.is_null() {
            return Err(std::io::Error::last_os_error());
        }

        let map_view = MapViewOfFile(map, FILE_MAP_READ, 0, 0, 0);

        if map_view.Value.is_null() {
            CloseHandle(map);
            return Err(std::io::Error::last_os_error());
        }

        Ok(std::slice::from_raw_parts(
            map_view.Value as *const u8,
            f.metadata().unwrap().len() as usize,
        ))
    }
}
