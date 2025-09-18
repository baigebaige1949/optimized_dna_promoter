"""条件扩散模型兼容性测试

验证条件扩散模型与现有diffusion_model.py的兼容性。
确保可以无缝切换两种模型。
"""

import torch
import sys
sys.path.append('..')

from optimized_dna_promoter.core.diffusion_model import DiffusionModel
from optimized_dna_promoter.core.conditional_diffusion_model import ConditionalDiffusionModel
from optimized_dna_promoter.config.model_config import DiffusionModelConfig, ConditionalDiffusionModelConfig
from optimized_dna_promoter.utils.logger import get_logger

logger = get_logger(__name__)


class CompatibilityTester:
    """兼容性测试器"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
    
    def test_basic_compatibility(self):
        """测试基础兼容性"""
        print("\n=== 基础兼容性测试 ===")
        
        # 创建基础扩散模型配置
        base_config = DiffusionModelConfig(
            hidden_dim=128,
            num_timesteps=100,
            unet_channels=[32, 64, 128]
        )
        
        # 创建条件扩散模型配置
        conditional_config = ConditionalDiffusionModelConfig(
            hidden_dim=128,
            num_timesteps=100,
            unet_channels=[32, 64, 128],
            condition_embedding_dim=128
        )
        
        try:
            # 创建模型
            base_model = DiffusionModel(base_config, vocab_size=4)
            conditional_model = ConditionalDiffusionModel(conditional_config, vocab_size=4)
            
            print(f"✓ 基础模型参数量: {sum(p.numel() for p in base_model.parameters()):,}")
            print(f"✓ 条件模型参数量: {sum(p.numel() for p in conditional_model.parameters()):,}")
            
            # 测试输入形状
            batch_size, channels, length = 2, 4, 100
            x = torch.randn(batch_size, channels, length)
            timesteps = torch.randint(0, 100, (batch_size,))
            
            # 基础模型前向传播
            with torch.no_grad():
                base_output = base_model(x, timesteps)
                print(f"✓ 基础模型输出形状: {base_output.shape}")
            
            # 条件模型前向传播（不带条件）
            with torch.no_grad():
                conditional_output = conditional_model(x, timesteps)
                print(f"✓ 条件模型输出形状: {conditional_output.shape}")
            
            # 条件模型前向传播（带条件）
            conditions = {
                'cell_type': torch.tensor([0, 1]),
                'temperature': torch.tensor([37.0, 30.0]),
                'ph': torch.tensor([7.0, 6.5])
            }
            
            with torch.no_grad():
                conditional_output_with_cond = conditional_model(x, timesteps, conditions)
                print(f"✓ 条件模型（带条件）输出形状: {conditional_output_with_cond.shape}")
            
            print("✓ 基础兼容性测试通过")
            
        except Exception as e:
            print(f"✗ 基础兼容性测试失败: {e}")
            return False
        
        return True
    
    def test_noise_scheduler_compatibility(self):
        """测试噪声调度器兼容性"""
        print("\n=== 噪声调度器兼容性测试 ===")
        
        try:
            config = DiffusionModelConfig(num_timesteps=100)
            
            # 创建两个模型
            base_model = DiffusionModel(config)
            conditional_model = ConditionalDiffusionModel(config)
            
            # 测试噪声调度器使用相同参数
            x_0 = torch.randn(2, 4, 50)
            timesteps = torch.tensor([10, 50])
            
            # 比较噪声添加结果
            base_noisy, base_noise = base_model.noise_scheduler.add_noise(x_0, timesteps)
            conditional_noisy, conditional_noise = conditional_model.noise_scheduler.add_noise(x_0, timesteps)
            
            # 验证结果一致性
            noise_diff = torch.abs(base_noisy - conditional_noisy).mean()
            print(f"✓ 噪声添加结果差异: {noise_diff.item():.6f}")
            
            if noise_diff < 1e-6:
                print("✓ 噪声调度器兼容性测试通过")
                return True
            else:
                print("✗ 噪声调度器结果不一致")
                return False
            
        except Exception as e:
            print(f"✗ 噪声调度器测试失败: {e}")
            return False
    
    def test_training_compatibility(self):
        """测试训练过程兼容性"""
        print("\n=== 训练过程兼容性测试 ===")
        
        try:
            config = ConditionalDiffusionModelConfig(
                hidden_dim=64,
                num_timesteps=50,
                unet_channels=[32, 64]
            )
            
            model = ConditionalDiffusionModel(config)
            model.train()
            
            # 模拟训练数据
            batch = {
                'sequences': torch.randn(4, 4, 100),
                'cell_type': torch.tensor([0, 1, 0, 1]),
                'temperature': torch.tensor([37.0, 30.0, 42.0, 25.0]),
                'ph': torch.tensor([7.0, 6.5, 7.5, 6.0])
            }
            
            # 训练步骤
            result = model.training_step(batch)
            
            print(f"✓ 训练损失: {result['loss'].item():.6f}")
            print(f"✓ 时间步形状: {result['timesteps'].shape}")
            print("✓ 训练过程兼容性测试通过")
            
            return True
            
        except Exception as e:
            print(f"✗ 训练过程测试失败: {e}")
            return False
    
    def test_conditional_features(self):
        """测试条件模型特有功能"""
        print("\n=== 条件模型特有功能测试 ===")
        
        try:
            config = ConditionalDiffusionModelConfig(
                hidden_dim=64,
                num_timesteps=50,
                unet_channels=[32, 64],
                use_classifier_free_guidance=True
            )
            
            model = ConditionalDiffusionModel(config)
            model.eval()
            
            # 测试条件验证器
            from optimized_dna_promoter.core.conditional_diffusion_model import ConditionValidator
            validator = ConditionValidator()
            
            partial_conditions = {'cell_type': torch.tensor([0, 1])}
            complete_conditions = validator.validate_and_complete(partial_conditions, batch_size=2)
            
            print(f"✓ 条件验证器补全条件: {list(complete_conditions.keys())}")
            
            # 测试采样
            with torch.no_grad():
                generated = model.sample(
                    shape=(2, 4, 50),
                    conditions=complete_conditions,
                    num_inference_steps=10,
                    guidance_scale=3.0
                )
                print(f"✓ 采样输出形状: {generated.shape}")
            
            # 测试无条件采样
            with torch.no_grad():
                generated_uncond = model.sample(
                    shape=(2, 4, 50),
                    conditions=None,
                    num_inference_steps=10
                )
                print(f"✓ 无条件采样输出形状: {generated_uncond.shape}")
            
            print("✓ 条件模型特有功能测试通过")
            return True
            
        except Exception as e:
            print(f"✗ 条件模型特有功能测试失败: {e}")
            return False
    
    def test_config_compatibility(self):
        """测试配置兼容性"""
        print("\n=== 配置兼容性测试 ===")
        
        try:
            # 测试配置继承
            from optimized_dna_promoter.config.model_config import ModelConfig
            
            config = ModelConfig()
            
            # 验证配置存在
            assert hasattr(config, 'diffusion'), "缺少diffusion配置"
            assert hasattr(config, 'conditional_diffusion'), "缺少conditional_diffusion配置"
            assert hasattr(config, 'predictor'), "缺少predictor配置"
            
            # 验证条件扩散配置继承基础配置
            assert config.conditional_diffusion.hidden_dim == config.diffusion.hidden_dim
            assert hasattr(config.conditional_diffusion, 'condition_embedding_dim')
            
            print(f"✓ 基础扩散配置: hidden_dim={config.diffusion.hidden_dim}")
            print(f"✓ 条件扩散配置: hidden_dim={config.conditional_diffusion.hidden_dim}, "
                  f"condition_embedding_dim={config.conditional_diffusion.condition_embedding_dim}")
            
            # 测试配置导出
            params = config.get_model_params()
            assert 'diffusion' in params
            assert 'conditional_diffusion' in params
            assert 'predictor' in params
            
            print("✓ 配置兼容性测试通过")
            return True
            
        except Exception as e:
            print(f"✗ 配置兼容性测试失败: {e}")
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        print("开始条件扩散模型兼容性测试")
        print("="*60)
        
        tests = [
            ('config_compatibility', self.test_config_compatibility),
            ('basic_compatibility', self.test_basic_compatibility),
            ('noise_scheduler_compatibility', self.test_noise_scheduler_compatibility),
            ('training_compatibility', self.test_training_compatibility),
            ('conditional_features', self.test_conditional_features)
        ]
        
        results = {}
        for test_name, test_func in tests:
            try:
                results[test_name] = test_func()
            except Exception as e:
                print(f"✗ {test_name} 测试异常: {e}")
                results[test_name] = False
        
        # 总结
        print("\n" + "="*60)
        print("测试结果总结:")
        
        passed_count = 0
        for test_name, passed in results.items():
            status = "✓ 通过" if passed else "✗ 失败"
            print(f"  {test_name}: {status}")
            if passed:
                passed_count += 1
        
        total_tests = len(results)
        print(f"\n总计: {passed_count}/{total_tests} 测试通过")
        
        if passed_count == total_tests:
            print("✓ 所有兼容性测试均通过！")
        else:
            print("✗ 部分测试失败，请检查代码")
        
        return passed_count == total_tests


def main():
    """主函数"""
    torch.manual_seed(42)
    
    tester = CompatibilityTester()
    success = tester.run_all_tests()
    
    return success


if __name__ == "__main__":
    main()
