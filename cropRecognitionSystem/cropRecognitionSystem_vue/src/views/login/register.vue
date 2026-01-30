<template>
	<div class="over">
		<div class="bottom">
			<div class="from">
				<div class="title">智慧农业农作物识别系统</div>
				<el-form :model="ruleForm" status-icon :rules="registerRules" ref="ruleFormRef">
					<el-form-item prop="username">
						<el-input prefix-icon="User" v-model="ruleForm.username" placeholder="请输入账号"></el-input>
					</el-form-item>
					<el-form-item prop="password">
						<el-input prefix-icon="Lock" type="password" v-model="ruleForm.password" autocomplete="off" placeholder="请输入密码"></el-input>
					</el-form-item>
					<el-form-item prop="confirm">
						<el-input prefix-icon="Lock" type="password" v-model="ruleForm.confirm" autocomplete="off" placeholder="请确认密码"></el-input>
					</el-form-item>
					<div style="margin: 0px 0px; margin-top: 20px">
						<el-button type="button" @click="submitForm(ruleFormRef)" style="width: 100%">注册</el-button>
					</div>
				</el-form>
			</div>
		</div>
	</div>
</template>

<script lang="ts" setup>
import { reactive, ref } from 'vue';
import type { FormInstance, FormRules } from 'element-plus';
import { ElMessage, ElMessageBox } from 'element-plus';
import router from '/@/router';
import request from '/@/utils/request';

const formSize = ref('default');
const ruleFormRef = ref<FormInstance>();

/*
 * 方法实现自定义校验，必须写在前面。
 */
const validatePass = (rule: any, value: any, callback: any) => {
	if (value === '') {
		callback(new Error('请输入密码'));
	} else {
		callback();
	}
};
const validatePass2 = (rule: any, value: any, callback: any) => {
	if (value === '') {
		callback(new Error('请再次输入密码'));
	} else if (value !== ruleForm.password) {
		callback(new Error('两次密码不一致!'));
	} else {
		callback();
	}
};

/*
 * 定义全局变量，等价Vue2中的data() return。
 */

const ruleForm = reactive({
	username: '',
	password: '',
	confirm: '',
	role: '',
});

/*
 * 校验规则。
 */
const registerRules = reactive<FormRules>({
	username: [
		{ required: true, message: '请输入账号', trigger: 'blur' },
		{ min: 3, max: 5, message: '长度在3-5个字符', trigger: 'blur' },
	],
	password: [
		{ validator: validatePass, trigger: 'blur' },
		{ min: 3, max: 5, message: '长度在3-5个字符', trigger: 'blur' },
	],
	confirm: [
		{ validator: validatePass2, trigger: 'blur' },
		{ min: 3, max: 5, message: '长度在3-5个字符', trigger: 'blur' },
	],
});

/*
 * 提交后的方法。
 */
const submitForm = (formEl: FormInstance | undefined) => {
	if (!formEl) return;
	formEl.validate((valid) => {
		if (valid) {
			console.log(ruleForm);
			request.post('/api/user/register', ruleForm).then((res) => {
				console.log(res);
				if (res.code == 0) {
					router.push('/login');
					ElMessage({
						type: 'success',
						message: '注册成功！',
					});
				} else {
					ElMessage({
						type: 'error',
						message: '用户名已存在！',
					});
					console.log(res.msg);
				}
			});
		} else {
			console.log('error submit!');
			return false;
		}
	});
};
</script>

<style lang="scss" scoped>
.over {
	display: flex;
	flex-direction: column;
	width: 100%;
	height: 100%;
	background-image: url('./bac.jpg');
	background-size: 100% 100%;
	background-size: cover;
	background-repeat: no-repeat;
	background-attachment: fixed; /*关键*/
	background-position: center;
}
.title {
	color: #409eff;
	display: flex;
	width: 100%;
	height: 20%;
	font-family: SourceHanSansSC;
	font-weight: 400;
	font-size: 29px;
	text-align: center;
	justify-content: center;
	align-items: center;
}
.bottom {
	display: flex;
	flex-direction: column;
	width: 100%;
	// height: 60%;
	align-items: center;
	justify-content: center;
	/* align-items: center; */
}
.from {
	width: 500px;
	height: 400px;
	color: #409eff;
	border-radius: 10px;
	font-size: 14px;
	padding: 0px 20px;
	text-align: center;
	line-height: 20px;
	font-weight: normal;
	font-style: normal;
	background: rgba(211, 211, 211, 0.7);
	display: flex;
	flex-direction: column;
	justify-content: center;
	margin-top: 150px;
}

.but {
	background: rgba(255, 255, 255, 0);
	color: #409eff;
	border: none;
	cursor: pointer;
}

:depp(.el-input__inner) {
	width: 80%;
}

.res {
	width: 100%;
	text-align: right;
	color: #409eff;
}
:deep(.register) {
	margin-top: 5px;
	padding: 0%;
}
</style>
