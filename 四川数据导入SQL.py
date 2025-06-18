import os
import psycopg2
import csv


def import_with_psycopg2():
    # 数据库配置
    db_config = {
        "host": "localhost",
        "port": 5432,
        "database": "wgl",
        "user": "postgres",
        "password": "072525"
    }

    # 数据目录
    data_dir = r"D:\Program data\critical-care-database-comprising-patients-with-infection-at-zigong-fourth-peoples-hospital-1.1\DataTables\DataTables"

    try:
        # 建立数据库连接
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()

        # 获取数据文件
        data_files = [f for f in os.listdir(data_dir)
                      if f.lower().startswith("dt")
                      and f.lower().endswith(".csv")]

        print("检测到的数据文件：")
        print(data_files)

        for file in data_files:
            file_path = os.path.join(data_dir, file)
            table_name = os.path.splitext(file)[0].lower()

            try:
                # 读取CSV文件头（跳过空列）
                with open(file_path, 'r', encoding='utf-8-sig') as f:
                    reader = csv.reader(f)
                    headers = [h.strip().lower().replace(" ", "_") for h in next(reader) if h.strip() != '']

                    # 创建表（先删除已存在的表）
                    cur.execute(f"DROP TABLE IF EXISTS {table_name}")
                    conn.commit()

                    # 构建创建表SQL（跳过空列）
                    create_sql = f"CREATE TABLE {table_name} ("
                    create_sql += " text, ".join([f'"{col}"' for col in headers]) + " text)"

                    cur.execute(create_sql)
                    conn.commit()

                    # 使用COPY导入数据（跳过第一列）
                    with open(file_path, 'r', encoding='utf-8-sig') as f:
                        # 创建临时文件跳过第一列
                        temp_file = f"temp_{file}"
                        with open(temp_file, 'w', encoding='utf-8-sig', newline='') as tf:
                            writer = csv.writer(tf)
                            for row in csv.reader(f):
                                writer.writerow(row[1:])  # 跳过第一列

                        # 导入处理后的文件
                        with open(temp_file, 'r', encoding='utf-8-sig') as tf:
                            cur.copy_expert(
                                f"COPY {table_name} FROM STDIN WITH CSV HEADER",
                                tf
                            )
                        os.remove(temp_file)  # 删除临时文件

                    conn.commit()
                    print(f"✅ 成功导入: {table_name}")

            except Exception as e:
                conn.rollback()
                print(f"❌ 导入失败: {file}")
                print(f"错误详情: {str(e)}")

                # 打印更多调试信息
                print("\n调试信息：")
                print(f"文件路径: {file_path}")
                print("前3行数据样本:")
                try:
                    with open(file_path, 'r', encoding='utf-8-sig') as f:
                        print([next(f) for _ in range(3)])
                except:
                    pass

            print("-" * 60)

    finally:
        # 确保连接关闭
        if 'conn' in locals():
            conn.close()


if __name__ == "__main__":
    print("=== 开始导入数据 ===")
    import_with_psycopg2()
    print("=== 导入完成 ===")