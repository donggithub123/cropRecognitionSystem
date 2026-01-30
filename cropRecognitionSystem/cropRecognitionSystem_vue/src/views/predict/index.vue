<template>
	<div class="system-predict-container layout-padding">
		<div class="system-predict-padding layout-padding-auto layout-padding-view">
			<div class="header">
				<div class="model">
					<el-select v-model="model" placeholder="请选择模型" size="large" style="width: 240px" @change="filterWeights">
						<el-option v-for="item in state.model_items" :key="item.value" :label="item.label" :value="item.value" />
					</el-select>
				</div>
				<div class="weight" style="margin-left: 20px">
					<el-select v-model="weight" placeholder="请选择权重" size="large" style="width: 240px">
						<el-option v-for="item in state.weight_item" :key="item.value" :label="item.label" :value="item.value" />
					</el-select>
				</div>
				<div class="button-section" style="margin-left: 20px">
					<el-button type="primary" @click="upData" class="predict-button">开始预测</el-button>
				</div>
			</div>
			<el-card shadow="hover" class="card">
				<el-upload
					v-model="state.form.image_path"
					ref="uploadFile"
					class="avatar-uploader"
					action="http://localhost:9999/files/upload"
					:show-file-list="false"
					:on-success="handleAvatarSuccessone"
				>
					<img v-if="imageUrl" :src="imageUrl" class="avatar" />
					<el-icon v-else class="avatar-uploader-icon"><Plus /></el-icon>
				</el-upload>
			</el-card>
			<el-card class="result-section" v-if="state.predictionResult.prediction">
				<div class="bottom">
					<div style="width: 33%">识别结果：{{ state.predictionResult.prediction }}</div>
					<div style="width: 33%">预测概率：{{ state.predictionResult.confidence }}</div>
					<div style="width: 33%">总时间：{{ state.predictionResult.total_time }}</div>
				</div>
			</el-card>
		</div>
	</div>
</template>


<script setup lang="ts" name="personal">
import { reactive, ref, onMounted } from 'vue';
import type { UploadInstance, UploadProps } from 'element-plus';
import { ElMessage } from 'element-plus';
import request from '/@/utils/request';
import { Plus } from '@element-plus/icons-vue';
import { useUserInfo } from '/@/stores/userInfo';
import { storeToRefs } from 'pinia';

const imageUrl = ref('');
const model = ref('');
const weight = ref('');
const uploadFile = ref<UploadInstance>();
const stores = useUserInfo();
const { userInfos } = storeToRefs(stores);
const state = reactive({
	model_items: [] as any,
	weight_items: [] as any,
	weight_item: [] as any,
	predictionResult: {
		prediction: '',
		confidence: '',
		total_time: '',
	},
	form: {
		username: '',
		image_path: null as any,
		weights: '',
		model_name: '',
	},
});

const handleAvatarSuccessone: UploadProps['onSuccess'] = (response, uploadFile) => {
	imageUrl.value = URL.createObjectURL(uploadFile.raw!);
	state.form.image_path = response.data;
};

const getData = () => {
	request.get('/api/flask/file_names').then((res) => {
		if (res.code == 0) {
			res.data = JSON.parse(res.data);
			state.model_items = res.data.model_items;
			state.weight_items = res.data.weight_items;
			state.weight_item = res.data.weight_items;
		} else {
			ElMessage.error(res.msg);
		}
	});
};

const filterWeights = () => {
	if (model.value) {
		state.weight_item = state.weight_items.filter((item) => item.label.includes(model.value));
	}
};

const upData = () => {
	if (!state.form.image_path) {
		ElMessage.error('请上传一张图片！');
		return;
	}
	state.form.weights = weight.value;
	state.form.model_name = model.value;
	state.form.username = userInfos.value.userName;
	console.log(state.form);
	request.post('/api/flask/predict', state.form).then((res) => {
		if (res.code == 0) {
			try {
				res.data = JSON.parse(res.data);
				state.predictionResult.prediction = res.data.prediction;
				state.predictionResult.confidence = res.data.confidence;
				state.predictionResult.total_time = res.data.total_time;
				console.log(state.predictionResult);
			} catch (error) {
				console.error('解析 JSON 时出错:', error);
			}
			ElMessage.success('预测成功！');
		} else {
			ElMessage.error(res.msg);
		}
	});
};

onMounted(() => {
	getData();
});
</script>

<style scoped lang="scss">
.system-predict-container {
	width: 100%;
	height: 100%;
	display: flex;
	flex-direction: column;
	.system-predict-padding {
		padding: 15px;
		.el-table {
			flex: 1;
		}
	}
}

.header {
	width: 100%;
	height: 5%;
	display: flex;
	justify-content: start;
	align-items: center;
	font-size: 20px;
}

.card {
	width: 100%;
	height: 800px;
	border-radius: 10px;
	margin-top: 15px;
	display: flex;
	justify-content: center;
	align-items: center;
}
.avatar-uploader .avatar {
	width: 100%;
	height: 600px;
	display: block;
}

.el-icon.avatar-uploader-icon {
	font-size: 28px;
	color: #8c939d;
	width: 100%;
	height: 600px;
	text-align: center;
}

.button-section {
	display: flex;
	justify-content: center;
}

.predict-button {
	width: 100%; /* 按钮宽度填满 */
}

.result-section {
	width: 100%;
	margin-top: 15px;
	text-align: center;
	display: flex; /* 添加 display: flex; */
	flex-direction: column;
	border-radius: 6px;
}
.bottom {
	width: 100%;
	font-size: 18px;
	display: flex; /* 添加 display: flex; */
	flex-direction: row;
	justify-content: center;
	align-items: center;
}
</style>
