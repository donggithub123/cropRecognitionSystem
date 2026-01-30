<template>
	<div class="over">
		<div class="bottom">
			<div class="from">
				<div class="title">智慧农业农作物识别系统</div>
				<el-form :model="ruleForm" status-icon :rules="registerRules" ref="ruleFormRef">
					<el-form-item prop="username">
						<el-input prefix-icon="User" v-model="ruleForm.username" placeholder="请输入账号"></el-input>
					</el-form-item>
					<el-form-item prop="pass">
						<el-input prefix-icon="Lock" type="password" v-model="ruleForm.password" autocomplete="off" placeholder="请输入密码"></el-input>
					</el-form-item>
					<div style="margin: 20px 0px">
						<el-button type="primary" @click="submitForm(ruleFormRef)" style="width: 100%">登录</el-button>
					</div>
					<div class="res" @click="router.push('/register')">
						<button class="but" type="button">前往注册 >></button>
					</div>
				</el-form>
			</div>
		</div>
	</div>
</template>

<script lang="ts" setup>
import { reactive, computed, ref } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import { ElMessage } from 'element-plus';
import { useI18n } from 'vue-i18n';
import Cookies from 'js-cookie';
import { storeToRefs } from 'pinia';
import { useThemeConfig } from '/@/stores/themeConfig';
import { initFrontEndControlRoutes } from '/@/router/frontEnd';
import { initBackEndControlRoutes } from '/@/router/backEnd';
import { Session } from '/@/utils/storage';
import { formatAxis } from '/@/utils/formatTime';
import { NextLoading } from '/@/utils/loading';
import type { FormInstance, FormRules } from 'element-plus';
import request from '/@/utils/request';

// 定义变量内容
const { t } = useI18n();
const storesThemeConfig = useThemeConfig();
const { themeConfig } = storeToRefs(storesThemeConfig);
const route = useRoute();
const router = useRouter();
const formSize = ref('default');
const ruleFormRef = ref<FormInstance>();

/*
 * 定义全局变量，等价Vue2中的data() return。
 */
const ruleForm = reactive({
	username: '',
	password: '',
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
		{ required: true, message: '请输入密码', trigger: 'blur' },
		{ min: 3, max: 5, message: '长度在3-5个字符', trigger: 'blur' },
	],
});

/*
 * 提交后的方法。
 */
// 时间获取
const currentTime = computed(() => {
	return formatAxis(new Date());
});
// 登录
const onSignIn = async () => {
	// 存储 token 到浏览器缓存
	Session.set('token', Math.random().toString(36).substr(0));
	// 模拟数据，对接接口时，记得删除多余代码及对应依赖的引入。用于 `/src/stores/userInfo.ts` 中不同用户登录判断（模拟数据）
	Cookies.set('userName', ruleForm.username);
	if (!themeConfig.value.isRequestRoutes) {
		// 前端控制路由，2、请注意执行顺序
		const isNoPower = await initFrontEndControlRoutes();
		signInSuccess(isNoPower);
	} else {
		// 模拟后端控制路由，isRequestRoutes 为 true，则开启后端控制路由
		// 添加完动态路由，再进行 router 跳转，否则可能报错 No match found for location with path "/"
		const isNoPower = await initBackEndControlRoutes();
		// 执行完 initBackEndControlRoutes，再执行 signInSuccess
		signInSuccess(isNoPower);
	}
};
// 登录成功后的跳转
const signInSuccess = (isNoPower: boolean | undefined) => {
	if (isNoPower) {
		ElMessage.warning('抱歉，您没有登录权限');
		Session.clear();
	} else {
		// 初始化登录成功时间问候语
		let currentTimeInfo = currentTime.value;
		// 登录成功，跳到转首页
		// 如果是复制粘贴的路径，非首页/登录页，那么登录成功后重定向到对应的路径中
		if (route.query?.redirect) {
			router.push({
				path: <string>route.query?.redirect,
				query: Object.keys(<string>route.query?.params).length > 0 ? JSON.parse(<string>route.query?.params) : '',
			});
		} else {
			router.push('/');
		}
		// 登录成功提示
		const signInText = t('message.signInText');
		ElMessage.success(`${currentTimeInfo}，${signInText}`);
		// 添加 loading，防止第一次进入界面时出现短暂空白
		NextLoading.start();
	}
};
const submitForm = (formEl: FormInstance | undefined) => {
	if (!formEl) return;
	formEl.validate((valid) => {
		if (valid) {
			request.post('/api/user/login', ruleForm).then((res) => {
				console.log(res);
				if (res.code == 0) {
					Cookies.set('role', res.data.role); //  设置角色
					//登录成功
					onSignIn();
				} else {
					ElMessage({
						type: 'error',
						message: res.msg,
					});
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
	background-position: center;
	background-size: 100% 100%;
	background-repeat: no-repeat;
}
.title {
	color: #409eff;
	display: flex;
	width: 100%;
	height: 30%;
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
	height: 350px;
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
	width: 60%;
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
