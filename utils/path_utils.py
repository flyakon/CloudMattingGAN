
def get_filename(file_path,is_suffix=True):
	file_name=file_path.replace('/','\\')
	file_name=file_name.split('\\')[-1]
	if is_suffix:
		return file_name
	else:
		index=file_name.rfind('.')
		if index>0:
			return file_name[0:index]
		else:
			return file_name